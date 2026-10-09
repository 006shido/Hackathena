"""Hash-pinned XSeg research executor; OpenCV parity required before adoption."""
from collections import Counter
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from ml.experiments.reference_backend.opencv_reference import ASSETS,fields
from ml.experiments.reference_backend.torch_reference import tensor,attribute

SHA='c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3'
class TorchOcclusion:
    OPERATORS={'Transpose','Conv','Mul','GlobalAveragePool','Add','Sqrt','Reciprocal','Max','Pad','Reshape','MatMul','ConvTranspose','Concat','Sigmoid'}
    def __init__(self,device='cuda'):
        self.device=torch.device(device);path=ASSETS/'xseg_1.onnx'
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=SHA:raise ValueError('XSeg checksum mismatch')
        graph=next(v for n,_,v in fields(memoryview(raw)) if n==7)
        self.constants={};self.nodes=[];inputs=[];outputs=[]
        for n,_,data in fields(graph):
            if n==5:
                name,value=tensor(data);self.constants[name]=value.to(self.device) if value.is_floating_point() else value
            elif n in (11,12):
                name=next(bytes(v).decode() for k,_,v in fields(data) if k==1)
                (inputs if n==11 else outputs).append(name)
            elif n==1:
                names=[];dest=[];attrs={};op=None
                for k,_,v in fields(data):
                    if k==1:names.append(bytes(v).decode())
                    elif k==2:dest.append(bytes(v).decode())
                    elif k==4:op=bytes(v).decode()
                    elif k==5:
                        key,value=attribute(v);attrs[key]=value
                if op not in self.OPERATORS or len(dest)!=1:raise ValueError(f'Unsupported node {op}')
                self.nodes.append((op,names,dest[0],attrs))
        if inputs!=['input'] or outputs!=['output']:raise ValueError('Unexpected XSeg interface')
        self.uses=Counter(name for _,names,_,_ in self.nodes for name in names if name)

    @torch.inference_mode()
    def __call__(self,bgr):
        array=np.asarray(bgr,dtype=np.float32)
        if array.shape!=(1,256,256,3) or not np.isfinite(array).all():raise ValueError('Expected finite NHWC BGR input')
        values={'input':torch.as_tensor(array,device=self.device)};uses=self.uses.copy()
        for op,names,out,a in self.nodes:
            x=[values[name] if name in values else self.constants[name] for name in names]
            if op=='Transpose':y=x[0].permute(a['perm'])
            elif op=='Conv':
                pads=a.get('pads',[0,0,0,0])
                if pads[:2]!=pads[2:] or a.get('auto_pad','NOTSET')!='NOTSET':raise ValueError('Unsupported convolution padding')
                y=F.conv2d(x[0],x[1],x[2] if len(x)>2 else None,a.get('strides',[1,1]),pads[:2],a.get('dilations',[1,1]),a.get('group',1))
            elif op=='ConvTranspose':
                pads=a.get('pads',[0,0,0,0])
                if a.get('auto_pad','NOTSET')!='NOTSET' or 'output_shape' in a:raise ValueError('Unsupported transposed convolution shape')
                y=F.conv_transpose2d(x[0],x[1],x[2] if len(x)>2 else None,stride=a.get('strides',[1,1]),output_padding=a.get('output_padding',[0,0]),groups=a.get('group',1),dilation=a.get('dilations',[1,1]))
                y=y[:,:,pads[0]:y.shape[2]-pads[2],pads[1]:y.shape[3]-pads[3]]
            elif op=='Mul':y=x[0]*x[1]
            elif op=='Add':y=x[0]+x[1]
            elif op=='GlobalAveragePool':y=x[0].mean(dim=tuple(range(2,x[0].ndim)),keepdim=True)
            elif op=='Sqrt':y=x[0].sqrt()
            elif op=='Reciprocal':y=x[0].reciprocal()
            elif op=='Max':
                y=x[0]
                for other in x[1:]:y=torch.maximum(y,other)
            elif op=='Pad':
                p=x[1].tolist()
                if len(p)!=8 or any(p[i] for i in [0,1,4,5]) or min(p)<0 or a.get('mode','constant')!='constant':raise ValueError('Unsupported pad')
                y=F.pad(x[0],[p[3],p[7],p[2],p[6]],value=float(x[2]) if len(x)>2 else 0)
            elif op=='Reshape':
                shape=x[1].tolist();shape=[x[0].shape[i] if d==0 else d for i,d in enumerate(shape)];y=x[0].reshape(shape)
            elif op=='MatMul':y=x[0]@x[1]
            elif op=='Concat':y=torch.cat(x,dim=a['axis'])
            elif op=='Sigmoid':y=x[0].sigmoid()
            else:raise AssertionError(op)
            values[out]=y
            for name in names:
                uses[name]-=1
                if uses[name]==0 and name in values and name!='output':del values[name]
        output=values['output']
        if output.numel()!=256*256 or not torch.isfinite(output).all():raise ValueError('Invalid XSeg output')
        return output.reshape(256,256)
