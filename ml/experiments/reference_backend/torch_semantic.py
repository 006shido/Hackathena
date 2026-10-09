"""Restricted hash-pinned semantic parser executor for offline research."""
from collections import Counter
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from ml.experiments.reference_backend.opencv_reference import ASSETS,fields
from ml.experiments.reference_backend.torch_reference import tensor,attribute

SHA='2218b6183c26ca5c83303232d682a536c670c13ea9695f716c777d1f244eefe9'
class TorchSemantic:
    OPS={'Shape','Constant','Gather','Conv','Relu','MaxPool','Add','AveragePool','Unsqueeze','Concat','Slice','Cast','Resize','Sigmoid','Mul'}
    def __init__(self,device='cuda'):
        self.device=torch.device(device);raw=(ASSETS/'bisenet_resnet_18.onnx').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=SHA:raise ValueError('Parser checksum mismatch')
        graph=next(v for n,_,v in fields(memoryview(raw)) if n==7);self.constants={};self.nodes=[];inputs=[];outputs=[]
        for n,_,data in fields(graph):
            if n==5:
                name,value=tensor(data);self.constants[name]=value.to(self.device) if value.is_floating_point() else value
            elif n in (11,12):
                name=next(bytes(v).decode() for k,_,v in fields(data) if k==1);(inputs if n==11 else outputs).append(name)
            elif n==1:
                names=[];dest=[];attrs={};op=next(bytes(v).decode() for k,_,v in fields(data) if k==4)
                for k,_,v in fields(data):
                    if k==1:names.append(bytes(v).decode())
                    elif k==2:dest.append(bytes(v).decode())
                    elif k==5:
                        if op=='Constant':
                            encoded=next(bytes(item) for num,_,item in fields(v) if num==5)
                            _,value=tensor(encoded+b'\x42\x00');attrs['value']=value.to(self.device) if value.is_floating_point() else value
                        else:
                            key,value=attribute(v);attrs[key]=value
                if op not in self.OPS or len(dest)!=1:raise ValueError(f'Unsupported parser node {op}')
                self.nodes.append((op,names,dest[0],attrs))
        if inputs!=['input'] or outputs!=['output','0','1']:raise ValueError('Unexpected parser interface')
        self.outputs=outputs;self.uses=Counter(name for _,names,_,_ in self.nodes for name in names if name)

    @torch.inference_mode()
    def __call__(self,array):
        array=np.asarray(array,dtype=np.float32)
        if array.shape!=(1,3,512,512) or not np.isfinite(array).all():raise ValueError('Expected finite normalized RGB NCHW input')
        values={'input':torch.as_tensor(array,device=self.device)};uses=self.uses.copy()
        for op,names,out,a in self.nodes:
            x=[(values[name] if name in values else self.constants[name]) if name else None for name in names]
            if op=='Constant':y=a['value']
            elif op=='Shape':y=torch.tensor(list(x[0].shape),dtype=torch.int64)
            elif op=='Gather':
                index=x[1].long();axis=a.get('axis',0);y=torch.index_select(x[0],axis,index.reshape(-1).to(x[0].device))
                if index.ndim==0:y=y.squeeze(axis)
                elif index.ndim!=1:raise ValueError('Unsupported Gather rank')
            elif op=='Conv':
                p=a.get('pads',[0,0,0,0])
                if p[:2]!=p[2:] or a.get('auto_pad','NOTSET')!='NOTSET':raise ValueError('Unsupported convolution pads')
                y=F.conv2d(x[0],x[1],x[2] if len(x)>2 else None,a.get('strides',[1,1]),p[:2],a.get('dilations',[1,1]),a.get('group',1))
            elif op=='Relu':y=F.relu(x[0])
            elif op in ('MaxPool','AveragePool'):
                p=a.get('pads',[0,0,0,0])
                if p[:2]!=p[2:]:raise ValueError('Unsupported pooling pads')
                if op=='MaxPool':y=F.max_pool2d(x[0],a['kernel_shape'],a.get('strides'),p[:2],a.get('dilations',[1,1]),bool(a.get('ceil_mode',0)))
                else:y=F.avg_pool2d(x[0],a['kernel_shape'],a.get('strides'),p[:2],bool(a.get('ceil_mode',0)),bool(a.get('count_include_pad',0)))
            elif op=='Add':y=x[0]+x[1]
            elif op=='Mul':y=x[0]*x[1]
            elif op=='Sigmoid':y=x[0].sigmoid()
            elif op=='Unsqueeze':
                axes=x[1].tolist() if len(x)>1 else a['axes'];y=x[0]
                for axis in sorted(axis if axis>=0 else axis+x[0].ndim+len(axes) for axis in axes):y=y.unsqueeze(axis)
            elif op=='Concat':y=torch.cat(x,dim=a['axis'])
            elif op=='Slice':
                slices=[slice(None)]*x[0].ndim;starts=x[1].tolist();ends=x[2].tolist();axes=x[3].tolist() if len(x)>3 else list(range(len(starts)));steps=x[4].tolist() if len(x)>4 else [1]*len(starts)
                for start,end,axis,step in zip(starts,ends,axes,steps):
                    if step<=0:raise ValueError('Unsupported slice step')
                    slices[axis]=slice(start,end,step)
                y=x[0][tuple(slices)]
            elif op=='Cast':
                if a['to']!=7:raise ValueError('Unsupported cast')
                y=x[0].long()
            elif op=='Resize':
                sizes=x[3].tolist()
                if sizes[:2]!=list(x[0].shape[:2]):raise ValueError('Unsupported nonspatial resize')
                if a.get('mode')=='nearest' and a.get('coordinate_transformation_mode')=='asymmetric' and a.get('nearest_mode')=='floor':y=F.interpolate(x[0],size=sizes[2:],mode='nearest')
                elif a.get('mode')=='linear' and a.get('coordinate_transformation_mode')=='align_corners':y=F.interpolate(x[0],size=sizes[2:],mode='bilinear',align_corners=True)
                else:raise ValueError('Unsupported resize semantics')
            else:raise AssertionError(op)
            values[out]=y
            for name in names:
                if not name:continue
                uses[name]-=1
                if uses[name]==0 and name in values and name not in self.outputs:del values[name]
        result=values['output']
        if result.shape!=(1,19,512,512) or not torch.isfinite(result).all():raise ValueError('Invalid semantic output')
        return result
