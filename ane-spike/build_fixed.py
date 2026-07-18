import struct, sys
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol
WPATH, OUT, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
def load(path):
    b=open(path,"rb").read(); assert b[0:8]==b"GAIARDR1"; c=8
    def u32():
        nonlocal c; v=struct.unpack_from("<I",b,c)[0]; c+=4; return v
    lc=u32(); u32(); u32(); L=[]
    for _ in range(lc):
        i=u32(); o=u32()
        w=np.frombuffer(b,dtype="<f4",count=i*o,offset=c).reshape(o,i).copy(); c+=i*o*4
        bs=np.frombuffer(b,dtype="<f4",count=o,offset=c).copy(); c+=o*4
        L.append((w,bs))
    return L
layers=load(WPATH); IN=23
@mb.program(input_specs=[mb.TensorSpec(shape=(N, IN))])
def net(x):
    v=x
    for i,(w,bs) in enumerate(layers):
        v=mb.linear(x=v,weight=w.astype(np.float32),bias=bs.astype(np.float32),name=f"linear_{i}")
        if i!=len(layers)-1: v=mb.relu(x=v,name=f"relu_{i}")
    return v
m=ct.convert(net,convert_to="mlprogram",compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL,minimum_deployment_target=ct.target.macOS15,
    inputs=[ct.TensorType(name="x",shape=(N,IN),dtype=np.float32)])
m.save(OUT); print("saved",OUT,"N=",N)
