"""Optional raw, machine-wide Windows paging counters for bounded benchmarks.

Differences of these cumulative values are counts, not instantaneous /sec rates.
They include other processes and must not be labelled as engine-only hard faults.
No sampling thread, privilege changes, cache eviction or memory allocation scan.
"""
import ctypes
from ctypes import wintypes
import sys


class PagingCounters:
    paths = {
        'page_reads': r'\Memory\Page Reads/sec',
        'pages_input': r'\Memory\Pages Input/sec',
        'pages_output': r'\Memory\Pages Output/sec',
        'transition_faults': r'\Memory\Transition Faults/sec',
        'disk_read_bytes': r'\PhysicalDisk(_Total)\Disk Read Bytes/sec',
    }

    def __init__(self):
        if sys.platform != 'win32':
            raise RuntimeError('Windows paging counters require Windows')
        self.dll = ctypes.WinDLL('pdh')
        self.query = wintypes.HANDLE()
        self.handles = {}
        self.dll.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        self.dll.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        self.dll.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
        self.dll.PdhCloseQuery.argtypes = [wintypes.HANDLE]
        class Raw(ctypes.Structure):
            _fields_ = [('status', wintypes.DWORD), ('time', wintypes.FILETIME),
                        ('first', ctypes.c_longlong), ('second', ctypes.c_longlong), ('multi', wintypes.DWORD)]
        self.Raw = Raw
        self.dll.PdhGetRawCounterValue.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(Raw)]
        self.check(self.dll.PdhOpenQueryW(None, 0, ctypes.byref(self.query)))
        try:
            for name, path in self.paths.items():
                handle = wintypes.HANDLE()
                self.check(self.dll.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(handle)))
                self.handles[name] = handle
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f'Windows performance counter error 0x{status & 0xffffffff:08x}')

    def sample(self):
        self.check(self.dll.PdhCollectQueryData(self.query))
        result = {}
        for name, handle in self.handles.items():
            raw, kind = self.Raw(), wintypes.DWORD()
            self.check(self.dll.PdhGetRawCounterValue(handle, ctypes.byref(kind), ctypes.byref(raw)))
            if raw.status not in (0, 1):
                self.check(raw.status)
            # These counters are cumulative rate counters (32- or 64-bit).
            if kind.value not in (0x10410400, 0x10410500):
                raise RuntimeError(f'Unexpected counter type for {name}: {kind.value:#x}')
            bits = 64 if kind.value & 0x100 else 32
            result[name] = {'value': raw.first & ((1 << bits) - 1), 'bits': bits}
        return result

    @staticmethod
    def difference(before, after):
        return {k: (v['value'] - before[k]['value']) % (1 << v['bits']) for k, v in after.items()}

    def close(self):
        if self.query:
            self.dll.PdhCloseQuery(self.query)
            self.query = wintypes.HANDLE()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
