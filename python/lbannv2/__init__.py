################################################################################
## Copyright 2014-2025 Lawrence Livermore National Security, LLC and other
## LBANN Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import sys
import torch

try:
    from .lib._lbannv2 import *
except ModuleNotFoundError:
    from .lib64._lbannv2 import *

# Initialize LBANNv2 library stuff. Among other things, this renames
# the PrivateUse1 backend to "lbann".
init_lbannv2()

torch._register_device_module("lbann", sys.modules[__name__])

unsupported_dtype = [
    torch.qint8,
    torch.qint32,
    torch.quint2x4,
    torch.quint4x2,
    torch.quint8,
]
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True,
    for_module=True,
    for_storage=True,
    unsupported_dtype=unsupported_dtype,
)


class MigratableMemory:
    """Use LBANNv2's allocator for the given device"""

    def __enter__(self):
        use_lbannv2_allocator_for(torch.device("cpu"))

    def __exit__(self, exc_type, exc_value, traceback):
        restore_default_allocator_for(torch.device("cpu"))


def make_migratory_tensor(ctor, *args, **kwargs):
    with MigratableMemory():
        return ctor(*args, **kwargs)
