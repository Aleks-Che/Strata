"""Read-only Windows/NVML observer independent of the pressure holder's lock.

Single-GPU diagnostic. Uses the installed System32 NVML, matching by PCI ID.
The owner must close/join the observer before hashing the JSONL artifact.
"""
import ctypes
import json
from pathlib import Path
import subprocess
import threading
import time
import traceback


def memory_within_limit(sample):
    for axis in ('ram','gpu'):
        total,free=sample[axis+'_total'],sample[axis+'_free']
        if type(total) is not int or type(free) is not int or total<=0 or not 0<=free<=total or free*20<total:
            return False
    total,free=sample['commit_total'],sample['commit_free']
    return type(total) is int and type(free) is int and total>0 and 0<=free<=total


class WindowsSampler:
    class RAM(ctypes.Structure):
        _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in
                 ['total','available','commit_total','commit_available','virtual_total','virtual_available','extended']]
    class GPU(ctypes.Structure):
        _fields_=[(n,ctypes.c_uint64) for n in ['total','free','used']]

    def __init__(self):
        self.initialized=False
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        self.kernel.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(self.RAM)]
        self.kernel.GlobalMemoryStatusEx.restype=ctypes.c_int
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=pci.bus_id','--format=csv,noheader'],
                                     text=True,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW).splitlines()
        if len(rows)!=1:raise RuntimeError('memory observer requires exactly one GPU')
        self.pci=rows[0].strip()
        self.nvml=ctypes.WinDLL('nvml.dll',winmode=0x800) # LOAD_LIBRARY_SEARCH_SYSTEM32
        self.nvml.nvmlInit_v2.argtypes=[];self.nvml.nvmlInit_v2.restype=ctypes.c_int
        self.nvml.nvmlShutdown.argtypes=[];self.nvml.nvmlShutdown.restype=ctypes.c_int
        self.nvml.nvmlDeviceGetHandleByPciBusId_v2.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_void_p)]
        self.nvml.nvmlDeviceGetHandleByPciBusId_v2.restype=ctypes.c_int
        self.nvml.nvmlDeviceGetMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(self.GPU)]
        self.nvml.nvmlDeviceGetMemoryInfo.restype=ctypes.c_int
        self._ok(self.nvml.nvmlInit_v2());self.initialized=True
        try:
            self.device=ctypes.c_void_p()
            self._ok(self.nvml.nvmlDeviceGetHandleByPciBusId_v2(self.pci.encode('ascii'),ctypes.byref(self.device)))
            if not self.device:raise RuntimeError('NVML returned no device')
        except BaseException:
            self.close();raise

    @staticmethod
    def _ok(code):
        if code:raise RuntimeError('NVML error '+str(code))

    def __call__(self):
        ram=self.RAM();ram.length=ctypes.sizeof(ram)
        if not self.kernel.GlobalMemoryStatusEx(ctypes.byref(ram)):
            raise ctypes.WinError(ctypes.get_last_error())
        gpu=self.GPU();self._ok(self.nvml.nvmlDeviceGetMemoryInfo(self.device,ctypes.byref(gpu)))
        return {'monotonic':time.monotonic(),'ram_total':ram.total,'ram_free':ram.available,
                'commit_total':ram.commit_total,'commit_free':ram.commit_available,
                'gpu_total':gpu.total,'gpu_free':gpu.free}

    def close(self):
        if self.initialized:
            self.initialized=False;self._ok(self.nvml.nvmlShutdown())


class MemoryObserver:
    def __init__(self,path,*,sampler=None,sequence=lambda:None,on_error=lambda:None,interval=.5):
        self.path=Path(path);self.sampler=sampler if sampler is not None else WindowsSampler()
        self.sequence,self.on_error,self.interval=sequence,on_error,interval
        self.stop=threading.Event();self.error=None;self.count=0
        self.thread=threading.Thread(target=self._run,daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            with self.path.open('x',encoding='utf-8') as f:
                while not self.stop.is_set():
                    if self.count>=20000:raise RuntimeError('memory observer sample limit')
                    sample={**self.sampler(),'native_sequence':self.sequence()}
                    f.write(json.dumps(sample,allow_nan=False)+'\n');f.flush();self.count+=1
                    if not memory_within_limit(sample):raise RuntimeError('independent global RAM/VRAM95 guard')
                    self.stop.wait(self.interval)
        except BaseException:
            self.error=traceback.format_exc()
            try:self.on_error()
            except BaseException:self.error+='\n'+traceback.format_exc()
        finally:
            try:self.sampler.close()
            except BaseException:self.error=(self.error or '')+'\n'+traceback.format_exc()

    def close(self):
        self.stop.set();self.thread.join(5)
        if self.thread.is_alive():raise RuntimeError('memory observer did not stop')

    def summary(self):
        return {'samples':self.count,'error':self.error,'stopped':not self.thread.is_alive(),
                'interval_seconds':self.interval,'pci_bus_id':getattr(self.sampler,'pci',None)}
