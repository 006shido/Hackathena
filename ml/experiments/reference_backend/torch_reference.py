"""Restricted executor for the hash-pinned research graph, using installed torch.

Not a general ONNX loader or a production backend. OpenCV remains the independent
numerical reference. Unexpected types/operators/resize semantics fail closed.
"""
from collections import Counter
import hashlib
import struct
import numpy as np
import torch
import torch.nn.functional as F
from ml.experiments.reference_backend.opencv_reference import fields, varint, ASSETS, SWAP_SHA


def integers(value):
    result=[]; offset=0
    while offset<len(value):
        item,offset=varint(value,offset)
        result.append(item if item<2**63 else item-2**64)
    return result


def attribute(data):
    name=None; value=None
    for number,wire,item in fields(data):
        if number==1:name=bytes(item).decode()
        elif number==2:value=struct.unpack('<f',item)[0]
        elif number==3:value=item if item<2**63 else item-2**64
        elif number==4:value=bytes(item).decode()
        elif number==8:
            if value is None:value=[]
            value.extend(integers(item) if wire==2 else [item if item<2**63 else item-2**64])
    if name is None or value is None:raise ValueError('Unsupported graph attribute')
    return name,value


def tensor(data):
    name=None; dims=[]; datatype=None; raw=None; values=[]
    for number,wire,item in fields(data):
        if number==1:dims.extend(integers(item) if wire==2 else [item])
        elif number==2:datatype=item
        elif number==8:name=bytes(item).decode()
        elif number==9:raw=item
        elif number==4:
            if wire!=2:raise ValueError('Unsupported unpacked float storage')
            raw=item
        elif number==7:values.extend(integers(item) if wire==2 else [item])
    if datatype not in (1,7) or name is None:raise ValueError('Unsupported initializer type')
    dtype='<f4' if datatype==1 else '<i8'
    array=np.frombuffer(raw,dtype=dtype).copy() if raw is not None else np.asarray(values,dtype=dtype)
    if array.size!=int(np.prod(dims)):raise ValueError('Invalid initializer dimensions')
    return name,torch.from_numpy(array.reshape(dims))


class TorchResearchSwapper:
    OPERATORS={'Conv','Pad','Relu','LeakyRelu','Add','Sub','Mul','Div','Sqrt',
               'ReduceMean','Unsqueeze','Slice','Gemm','Resize','Tanh'}

    def __init__(self,device='cuda'):
        self.device=torch.device(device)
        path=ASSETS/'inswapper_128.onnx'
        with path.open('rb') as file:
            if hashlib.file_digest(file,'sha256').hexdigest()!=SWAP_SHA:raise ValueError('Research graph digest mismatch')
        graph=next(value for number,_,value in fields(memoryview(path.read_bytes())) if number==7)
        self.constants={};self.nodes=[];self.outputs=[];self.inputs=[]
        for number,_,data in fields(graph):
            if number==5:
                name,value=tensor(data)
                # Integer initializers only control shapes/slices in this graph.
                self.constants[name]=value.to(self.device) if value.is_floating_point() else value
            elif number in (11,12):
                name=next(bytes(v).decode() for n,_,v in fields(data) if n==1)
                (self.inputs if number==11 else self.outputs).append(name)
            elif number==1:
                inputs=[];outputs=[];attrs={};op=None
                for n,_,v in fields(data):
                    if n==1:inputs.append(bytes(v).decode())
                    elif n==2:outputs.append(bytes(v).decode())
                    elif n==4:op=bytes(v).decode()
                    elif n==5:
                        key,value=attribute(v);attrs[key]=value
                if op not in self.OPERATORS or len(outputs)!=1:raise ValueError(f'Unsupported research node: {op}')
                self.nodes.append((op,inputs,outputs[0],attrs))
        if self.inputs!=['target','source'] or self.outputs!=['output']:raise ValueError('Unexpected research graph interface')
        self.uses=Counter(name for _,names,_,_ in self.nodes for name in names if name)

    @torch.inference_mode()
    def __call__(self,target,source):
        values={'target':torch.as_tensor(target,dtype=torch.float32,device=self.device),
                'source':torch.as_tensor(source,dtype=torch.float32,device=self.device)}
        if values['target'].shape!=(1,3,128,128) or values['source'].shape!=(1,512):raise ValueError('Unexpected research input shape')
        uses=self.uses.copy()
        for op,names,out,a in self.nodes:
            x=[(values[name] if name in values else self.constants[name]) if name else None for name in names]
            if op=='Conv':
                pads=a.get('pads',[0,0,0,0])
                if pads[0]!=pads[2] or pads[1]!=pads[3]:raise ValueError('Asymmetric convolution unsupported')
                result=F.conv2d(x[0],x[1],x[2] if len(x)>2 else None,stride=a.get('strides',[1,1]),
                    padding=pads[:2],dilation=a.get('dilations',[1,1]),groups=a.get('group',1))
            elif op=='Pad':
                pads=x[1].tolist()
                if len(pads)!=8 or any(pads[i] for i in [0,1,4,5]) or a.get('mode')!='reflect':raise ValueError('Unexpected padding')
                result=F.pad(x[0],[pads[3],pads[7],pads[2],pads[6]],mode='reflect')
            elif op=='Relu':result=F.relu(x[0])
            elif op=='LeakyRelu':result=F.leaky_relu(x[0],a.get('alpha',.01))
            elif op=='Add':result=x[0]+x[1]
            elif op=='Sub':result=x[0]-x[1]
            elif op=='Mul':result=x[0]*x[1]
            elif op=='Div':result=x[0]/x[1]
            elif op=='Sqrt':result=torch.sqrt(x[0])
            elif op=='Tanh':result=torch.tanh(x[0])
            elif op=='ReduceMean':result=x[0].mean(dim=tuple(a['axes']),keepdim=bool(a.get('keepdims',1)))
            elif op=='Unsqueeze':
                result=x[0]
                axes=[axis if axis>=0 else axis+x[0].ndim+len(a['axes']) for axis in a['axes']]
                for axis in sorted(axes):result=result.unsqueeze(axis)
            elif op=='Slice':
                starts,ends=x[1].tolist(),x[2].tolist()
                axes=x[3].tolist() if len(x)>3 else list(range(len(starts)))
                steps=x[4].tolist() if len(x)>4 else [1]*len(starts)
                slices=[slice(None)]*x[0].ndim
                for start,end,axis,step in zip(starts,ends,axes,steps):
                    if step<=0:raise ValueError('Unsupported slice step')
                    slices[axis]=slice(start,end,step)
                result=x[0][tuple(slices)]
            elif op=='Gemm':
                left=x[0].T if a.get('transA',0) else x[0]
                right=x[1].T if a.get('transB',0) else x[1]
                result=a.get('alpha',1)*(left@right)
                if len(x)>2:result=result+a.get('beta',1)*x[2]
            elif op=='Resize':
                if a.get('mode')!='linear' or a.get('coordinate_transformation_mode')!='pytorch_half_pixel':raise ValueError('Unexpected resize semantics')
                scales=x[2].tolist()
                if len(scales)!=4 or scales[:2]!=[1.,1.]:raise ValueError('Unexpected resize scales')
                result=F.interpolate(x[0],scale_factor=tuple(scales[2:]),mode='bilinear',align_corners=False)
            else:raise AssertionError(op)
            values[out]=result
            for name in names:
                if not name:continue
                uses[name]-=1
                if uses[name]==0 and name in values and name not in self.outputs:del values[name]
        output=values['output']
        if output.shape!=(1,3,128,128) or not torch.isfinite(output).all():raise ValueError('Invalid research output')
        return output
