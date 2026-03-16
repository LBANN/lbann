import torch
from typing import Callable, Union

try:
    from .lib._lbannv2 import migrate
except ModuleNotFoundError:
    from .lib64._lbannv2 import migrate


def automigrate(f: Union[Callable, torch.fx.GraphModule]) -> torch.fx.GraphModule:
    """Check the graph for candidates for automatic pointer migration,
    replacing them with appropriate calls to 'migrate'. This function
    operates at the ATen IR (FX Graph) level, so it cannot perfectly
    determine all cases in which a migrate is possible. Symbolic
    tracing cannot, for instance, tell the device on which inputs or
    "member tensors" (e.g., of some nn layer) reside. We can make some
    inferences, though (e.g., all nodes downstream of a memory
    relocation call ("to", "cpu", etc) can be assumed to live on that
    device until the next such relocation call). Additionally, we
    cannot, in general, know the provenance of the underlying memory
    of a given tensor. At this time, we only support migration that
    comes from LBANNv2 allocators, as we are 100% sure of its
    allocation. We could likely support any memory that was allocated
    by or registered with the HIP runtime, though -- "future work".

    Args:
        f (Union[Callable, torch.fx.GraphModule]): Any callable
            amenable to symbolic_trace()-ing. If this is a
            torch.fx.GraphModule, it will be modified in-place and
            returned.

    Returns:
        A torch.fx.GraphModule representing the input Callable, with
            "data movement" nodes replaced with LBANNv2 pointer
            "migration", when appropriate. If the input was already a
            torch.fx.GraphModule, it is modified in-place and
            returned.

    """

    def safe_for_migrate(n: torch.fx.graph.Node) -> bool:
        """
        At the IR level, a tensor is a candidate for migration if
        it isn't used multiple places and if the underlying operation
        isn't trying to change more than just the device.
        """
        input_ok = len(n.args[0].users) == 1
        args_ok = len(node.kwargs) == 1 and "device" in node.kwargs
        # If we're dealing with 'cuda' or 'cpu', then it's ok for the
        # kwargs to be empty (note that 'cuda' also supports 'device'
        # as a kwarg).
        if n.target != "to":
            args_ok = args_ok or len(node.kwargs) == 0
        return input_ok and args_ok

    def get_target_device(n: torch.fx.graph.Node) -> torch.device:
        return (
            torch.device(n.kwargs["device"])
            if "device" in n.kwargs
            else torch.device(str(n.target))
        )

    if isinstance(f, torch.fx.GraphModule):
        gm = f
    else:
        gm = torch.fx.symbolic_trace(f)

    # We can handle "to" or the device-specific methods ("cuda", e.g.).
    migrate_candidates = ["to", "cuda", "cpu"]
    for node in gm.graph.nodes:
        if node.target in migrate_candidates and safe_for_migrate(node):
            with gm.graph.inserting_before(node):
                # Add a new node
                new_node = gm.graph.call_function(
                    migrate,
                    args=(*node.args, get_target_device(node)),
                )
                node.replace_all_uses_with(new_node)

            gm.graph.erase_node(node)

    gm.recompile()
    return gm
