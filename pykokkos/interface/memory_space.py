from enum import Enum

from pykokkos.bindings import kokkos
from pykokkos.kokkos_manager import is_uvm_enabled

from .execution_space import ExecutionSpace


class MemorySpace(Enum):
    CudaUVMSpace = kokkos.CudaUVMSpace
    CudaSpace = kokkos.CudaSpace
    HIPSpace = kokkos.HIPSpace
    HIPHostPinnedSpace = kokkos.HIPHostPinnedSpace
    OpenMPTargetSpace = kokkos.OpenMPTargetSpace
    HostSpace = kokkos.HostSpace
    MemorySpaceDefault = None


def get_default_memory_space(space: ExecutionSpace) -> MemorySpace:
    """
    Map from execution spaces to default memory spaces

    :param space: the execution space
    :returns: the default memory space
    """

    if space is ExecutionSpace.Cuda:
        if is_uvm_enabled():
            return MemorySpace.CudaUVMSpace
        else:
            return MemorySpace.CudaSpace
    elif space is ExecutionSpace.HIP:
        if is_uvm_enabled():
            return MemorySpace.HIPHostPinnedSpace
        else:
            return MemorySpace.HIPSpace
    else:
        return MemorySpace.HostSpace
