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

from ._automigrate import automigrate

# Setup state needed by the library
init_lbannv2()

def is_available():
    try:
        return bool(is_lbannv2_gpu_available())
    except Exception:
        return False

class MigratableMemory:
    """Use LBANNv2's allocator for the given device"""

    def __enter__(self):
        use_mi300a_host_allocator()

    def __exit__(self, exc_type, exc_value, traceback):
        use_pytorch_host_allocator()


def make_migratory_tensor(ctor, *args, **kwargs):
    with MigratableMemory():
        return ctor(*args, **kwargs)
