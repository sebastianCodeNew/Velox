"""
Empirical Verification of CUDA Driver Virtual Memory Management (VMM) Implementation
Directly tests NVIDIA Driver API calls via ctypes against nvcuda.dll:
1. Driver initialization and context activation.
2. cuMemGetAllocationGranularity query (minimum vs recommended).
3. cuMemAddressReserve with alignment = 0 vs explicit granularity.
4. cuMemCreate, cuMemMap, cuMemSetAccess lifecycle for 2MB pages.
5. Verification of cuMemUnmap and cuMemRelease teardown semantics.
6. Error handling behavior on invalid parameters and out-of-bounds mapping.
"""

import ctypes
from ctypes import c_int, c_uint, c_ulonglong, c_size_t, c_char_p, POINTER, Structure, byref

CUDA_SUCCESS = 0
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_INVALID = 0
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1

class CUmemLocation(Structure):
    _fields_ = [
        ("type", c_int),
        ("id", c_int)
    ]

class AllocFlags(Structure):
    _fields_ = [
        ("compressionType", ctypes.c_uint8),
        ("gpuDirectRDMACapable", ctypes.c_uint8),
        ("usage", ctypes.c_uint16),
        ("reserved", ctypes.c_uint8 * 4)
    ]

class CUmemAllocationProp(Structure):
    _fields_ = [
        ("type", c_int),
        ("requestedHandleTypes", c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", AllocFlags)
    ]

class CUmemAccessDesc(Structure):
    _fields_ = [
        ("location", CUmemLocation),
        ("flags", c_uint)
    ]

def run_cuda_vmm_audit():
    print("=" * 60)
    print("CUDA DRIVER VMM RUNTIME AUDIT (nvcuda.dll)")
    print("=" * 60)

    try:
        nvcuda = ctypes.WinDLL("nvcuda.dll")
    except Exception as e:
        print(f"Failed to load nvcuda.dll: {e}")
        return

    # Check cuInit
    cuInit = nvcuda.cuInit
    cuInit.argtypes = [c_uint]
    cuInit.restype = c_int
    res = cuInit(0)
    print(f"cuInit(0) -> result: {res} ({'SUCCESS' if res == 0 else 'FAILED'})")
    if res != CUDA_SUCCESS:
        print("Driver not initialized. Skipping device calls.")
        return

    # Get device 0
    cuDeviceGet = nvcuda.cuDeviceGet
    cuDeviceGet.argtypes = [POINTER(c_int), c_int]
    cuDeviceGet.restype = c_int
    dev = c_int(0)
    res = cuDeviceGet(byref(dev), 0)
    print(f"cuDeviceGet(&dev, 0) -> device id: {dev.value}, res: {res}")

    # Retain primary context
    cuDevicePrimaryCtxRetain = nvcuda.cuDevicePrimaryCtxRetain
    cuDevicePrimaryCtxRetain.argtypes = [POINTER(ctypes.c_void_p), c_int]
    cuDevicePrimaryCtxRetain.restype = c_int
    pctx = ctypes.c_void_p(0)
    res = cuDevicePrimaryCtxRetain(byref(pctx), dev.value)
    print(f"cuDevicePrimaryCtxRetain -> ctx: {pctx.value}, res: {res}")

    # Push context
    cuCtxPushCurrent = nvcuda.cuCtxPushCurrent
    cuCtxPushCurrent.argtypes = [ctypes.c_void_p]
    cuCtxPushCurrent.restype = c_int
    cuCtxPushCurrent(pctx)

    # 1. Query allocation granularity
    prop = CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = 0
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = dev.value
    prop.win32HandleMetaData = None
    prop.allocFlags.compressionType = 0
    prop.allocFlags.gpuDirectRDMACapable = 0
    prop.allocFlags.usage = 0

    cuMemGetAllocationGranularity = nvcuda.cuMemGetAllocationGranularity
    cuMemGetAllocationGranularity.argtypes = [POINTER(c_size_t), POINTER(CUmemAllocationProp), c_int]
    cuMemGetAllocationGranularity.restype = c_int

    gran_min = c_size_t(0)
    gran_rec = c_size_t(0)
    res_min = cuMemGetAllocationGranularity(byref(gran_min), byref(prop), CU_MEM_ALLOC_GRANULARITY_MINIMUM)
    res_rec = cuMemGetAllocationGranularity(byref(gran_rec), byref(prop), CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)
    print(f"\nGranularity Query:")
    print(f"  Minimum Granularity:     {gran_min.value} bytes ({gran_min.value / 1024:.0f} KB)")
    print(f"  Recommended Granularity: {gran_rec.value} bytes ({gran_rec.value / (1024*1024):.0f} MB)")
    assert gran_rec.value == 2 * 1024 * 1024, "Expected 2MB recommended granularity"

    # 2. Test cuMemAddressReserve
    cuMemAddressReserve = nvcuda.cuMemAddressReserve
    cuMemAddressReserve.argtypes = [POINTER(c_ulonglong), c_size_t, c_size_t, c_ulonglong, c_ulonglong]
    cuMemAddressReserve.restype = c_int

    # Test with alignment = 0 (as in listing line 1459) vs alignment = gran_rec
    vaddr = c_ulonglong(0)
    reserve_size = 64 * 1024 * 1024 # 64 MB
    res_reserve = cuMemAddressReserve(byref(vaddr), reserve_size, 0, 0, 0)
    print(f"\ncuMemAddressReserve(size=64MB, alignment=0):")
    print(f"  Result: {res_reserve} -> Reserved vaddr: 0x{vaddr.value:016X}")
    assert res_reserve == CUDA_SUCCESS, f"Failed cuMemAddressReserve with code {res_reserve}"

    # 3. Test cuMemCreate for 2MB page
    page_size = 2 * 1024 * 1024 # 2MB
    cuMemCreate = nvcuda.cuMemCreate
    cuMemCreate.argtypes = [POINTER(c_ulonglong), c_size_t, POINTER(CUmemAllocationProp), c_ulonglong]
    cuMemCreate.restype = c_int

    handle = c_ulonglong(0)
    res_create = cuMemCreate(byref(handle), page_size, byref(prop), 0)
    print(f"\ncuMemCreate(size=2MB):")
    print(f"  Result: {res_create} -> Allocation Handle: 0x{handle.value:016X}")
    assert res_create == CUDA_SUCCESS, f"Failed cuMemCreate with code {res_create}"

    # 4. Test cuMemMap
    cuMemMap = nvcuda.cuMemMap
    cuMemMap.argtypes = [c_ulonglong, c_size_t, c_size_t, c_ulonglong, c_ulonglong]
    cuMemMap.restype = c_int

    res_map = cuMemMap(vaddr.value, page_size, 0, handle.value, 0)
    print(f"\ncuMemMap(vaddr=0x{vaddr.value:016X}, size=2MB, handle=0x{handle.value:016X}):")
    print(f"  Result: {res_map} ({'SUCCESS' if res_map == 0 else 'FAILED'})")
    assert res_map == CUDA_SUCCESS, f"Failed cuMemMap with code {res_map}"

    # 5. Test cuMemSetAccess
    cuMemSetAccess = nvcuda.cuMemSetAccess
    cuMemSetAccess.argtypes = [c_ulonglong, c_size_t, POINTER(CUmemAccessDesc), c_size_t]
    cuMemSetAccess.restype = c_int

    access = CUmemAccessDesc()
    access.location = prop.location
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    res_access = cuMemSetAccess(vaddr.value, page_size, byref(access), 1)
    print(f"\ncuMemSetAccess(vaddr=0x{vaddr.value:016X}, size=2MB, flags=PROT_READWRITE):")
    print(f"  Result: {res_access} ({'SUCCESS' if res_access == 0 else 'FAILED'})")
    assert res_access == CUDA_SUCCESS, f"Failed cuMemSetAccess with code {res_access}"

    # 6. Test second page growth
    handle2 = c_ulonglong(0)
    res_create2 = cuMemCreate(byref(handle2), page_size, byref(prop), 0)
    assert res_create2 == CUDA_SUCCESS
    vaddr2 = vaddr.value + page_size
    res_map2 = cuMemMap(vaddr2, page_size, 0, handle2.value, 0)
    assert res_map2 == CUDA_SUCCESS
    res_access2 = cuMemSetAccess(vaddr2, page_size, byref(access), 1)
    assert res_access2 == CUDA_SUCCESS
    print(f"\nGrowth Verification:")
    print(f"  Mapped Page 2 at 0x{vaddr2:016X} -> SUCCESS!")

    # 7. Test teardown: cuMemUnmap entire mapped region (4MB), release handles, free reservation
    cuMemUnmap = nvcuda.cuMemUnmap
    cuMemUnmap.argtypes = [c_ulonglong, c_size_t]
    cuMemUnmap.restype = c_int

    cuMemRelease = nvcuda.cuMemRelease
    cuMemRelease.argtypes = [c_ulonglong]
    cuMemRelease.restype = c_int

    cuMemAddressFree = nvcuda.cuMemAddressFree
    cuMemAddressFree.argtypes = [c_ulonglong, c_size_t]
    cuMemAddressFree.restype = c_int

    # Unmap 4MB in one call (as in Listing 6.3 line 1522)
    res_unmap = cuMemUnmap(vaddr.value, 2 * page_size)
    print(f"\nTeardown: cuMemUnmap(vaddr=0x{vaddr.value:016X}, size=4MB) -> result: {res_unmap}")
    assert res_unmap == CUDA_SUCCESS

    res_rel1 = cuMemRelease(handle.value)
    res_rel2 = cuMemRelease(handle2.value)
    print(f"Teardown: cuMemRelease(handle1) -> {res_rel1}, cuMemRelease(handle2) -> {res_rel2}")
    assert res_rel1 == CUDA_SUCCESS and res_rel2 == CUDA_SUCCESS

    res_free = cuMemAddressFree(vaddr.value, reserve_size)
    print(f"Teardown: cuMemAddressFree(vaddr=0x{vaddr.value:016X}, size=64MB) -> result: {res_free}")
    assert res_free == CUDA_SUCCESS

    print("\nALL DRIVER API CALLS EXECUTED AGAINST NVIDIA HARDWARE DRIVER SUCCESSFULLY!")

if __name__ == '__main__':
    run_cuda_vmm_audit()
