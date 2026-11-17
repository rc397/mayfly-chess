"""The graphics cards Windows reports, in DXGI's order, which is the device number DirectML uses. No packages: ctypes
calls CreateDXGIFactory1 and EnumAdapters1 the way a C program would.

  python chess/volunteer/gpus.py
"""
import ctypes
from ctypes import wintypes

DXGI_ADAPTER_FLAG_SOFTWARE = 2


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [("Description", wintypes.WCHAR * 128), ("VendorId", wintypes.UINT), ("DeviceId", wintypes.UINT),
                ("SubSysId", wintypes.UINT), ("Revision", wintypes.UINT),
                ("DedicatedVideoMemory", ctypes.c_size_t), ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t), ("AdapterLuid", LUID), ("Flags", wintypes.UINT)]


class GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]


# IID_IDXGIFactory1 {770aae78-f26f-4dba-a829-253c83d1b387}
IID_FACTORY1 = GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
VENDORS = {0x10DE: "NVIDIA", 0x1002: "AMD", 0x8086: "Intel", 0x1414: "Microsoft"}


def _method(obj, index, restype, *argtypes):
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])


def list_gpus():
    """[{"index", "name", "vendor", "vram_gb"}] for the real cards (software renderers left out)."""
    factory = ctypes.c_void_p()
    if ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(IID_FACTORY1), ctypes.byref(factory)) != 0:
        return []
    out = []
    try:
        enum1 = _method(factory, 12, ctypes.HRESULT, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            try:
                enum1(factory, i, ctypes.byref(adapter))
            except OSError:                   # DXGI_ERROR_NOT_FOUND: no more adapters
                break
            desc = DXGI_ADAPTER_DESC1()
            _method(adapter, 10, ctypes.HRESULT, ctypes.POINTER(DXGI_ADAPTER_DESC1))(adapter, ctypes.byref(desc))
            _method(adapter, 2, wintypes.ULONG)(adapter)                       # Release
            if not desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE:
                out.append({"index": i, "name": desc.Description.strip(),
                            "vendor": VENDORS.get(desc.VendorId, hex(desc.VendorId)),
                            "vram_gb": round(desc.DedicatedVideoMemory / 2 ** 30, 1)})
            i += 1
    finally:
        _method(factory, 2, wintypes.ULONG)(factory)
    return out


if __name__ == "__main__":
    for g in list_gpus():
        print(f"{g['index']}: {g['name']} ({g['vendor']}, {g['vram_gb']} GB)")
