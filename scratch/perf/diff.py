import struct, zlib, sys
def readpng(p):
    d=open(p,'rb').read(); assert d[:8]==b'\x89PNG\r\n\x1a\n'; i=8; w=h=0; idat=b''
    while i<len(d):
        ln=struct.unpack('>I',d[i:i+4])[0]; t=d[i+4:i+8]; c=d[i+8:i+8+ln]
        if t==b'IHDR': w,h=struct.unpack('>II',c[:8])
        elif t==b'IDAT': idat+=c
        i+=12+ln
    raw=zlib.decompress(idat); bpp=3; stride=w*bpp; out=bytearray(); prev=bytearray(stride)
    pos=0
    for y in range(h):
        f=raw[pos]; pos+=1; line=bytearray(raw[pos:pos+stride]); pos+=stride
        for x in range(stride):
            a=line[x-bpp] if x>=bpp else 0; b=prev[x]; cc=prev[x-bpp] if x>=bpp else 0
            if f==1: line[x]=(line[x]+a)&255
            elif f==2: line[x]=(line[x]+b)&255
            elif f==3: line[x]=(line[x]+((a+b)>>1))&255
            elif f==4:
                p2=a+b-cc; pa=abs(p2-a); pb=abs(p2-b); pc=abs(p2-cc)
                pr=a if pa<=pb and pa<=pc else (b if pb<=pc else cc); line[x]=(line[x]+pr)&255
        out+=line; prev=line
    return w,h,bytes(out)
w,h,a=readpng(sys.argv[1]); _,_,b=readpng(sys.argv[2])
mx=0; n=0; s=0
for i in range(len(a)):
    dv=abs(a[i]-b[i]);
    if dv>mx: mx=dv
    s+=dv; n+= (dv>0)
print(f"max byte diff={mx}  nonzero bytes={n}/{len(a)}  mean={s/len(a):.6f}")
