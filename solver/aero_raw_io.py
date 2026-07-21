"""Aero raw tensor IO — shared Python<->C++ boundary format.
Layout: magic b'AERO' | int32 ndim | int32 dims[ndim] | float32 data (C-order).
"""
import numpy as np, struct
def write_raw(path, arr):
    arr=np.ascontiguousarray(np.asarray(arr,dtype=np.float32))
    with open(path,'wb') as f:
        f.write(b'AERO'); f.write(struct.pack('<i',arr.ndim))
        f.write(struct.pack('<%di'%arr.ndim,*arr.shape)); f.write(arr.tobytes())
def read_raw(path):
    with open(path,'rb') as f:
        assert f.read(4)==b'AERO', "bad magic"
        nd=struct.unpack('<i',f.read(4))[0]
        dims=struct.unpack('<%di'%nd,f.read(4*nd))
        return np.frombuffer(f.read(),dtype='<f4').reshape(dims).copy()
