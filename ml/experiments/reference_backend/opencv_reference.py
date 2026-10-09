"""Offline official InsightFace benchmark using existing OpenCV only.

Weights are research assets, not a production replacement. No package install
or project-environment mutation is needed. Pixel synthesis runs on CPU.
"""
from pathlib import Path
import hashlib
import cv2
import numpy as np
from ml.training.face_preprocessing import estimate_umeyama_similarity_transform

ASSETS=Path(__file__).parent/'assets'
SWAP_SHA='e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af'
RECOGNITION_SHA='4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43'
CANONICAL=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
    [41.5493,92.3655],[70.7299,92.2041]],dtype=np.float32)


def varint(data,offset):
    value=0
    for shift in range(0,70,7):
        if offset>=len(data):raise ValueError('Truncated protobuf integer')
        byte=data[offset];offset+=1
        value|=(byte&127)<<shift
        if byte<128:return value,offset
    raise ValueError('Oversized protobuf integer')


def fields(data):
    offset=0
    while offset<len(data):
        tag,offset=varint(data,offset);number,wire=tag>>3,tag&7
        if number==0:raise ValueError('Invalid protobuf field')
        if wire==0:
            value,offset=varint(data,offset)
        elif wire in (1,2,5):
            if wire==2:length,offset=varint(data,offset)
            else:length=8 if wire==1 else 4
            if offset+length>len(data):raise ValueError('Truncated protobuf field')
            value=data[offset:offset+length];offset+=length
        else:raise ValueError('Unsupported protobuf wire type')
        yield number,wire,value


def read_embedding_map(path):
    # Only read the fixed official FLOAT initializer and input names. The ONNX
    # graph is executed by OpenCV; this is not a generic model interpreter.
    # Schema: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
    data=memoryview(path.read_bytes())
    graph=next(value for number,_,value in fields(data) if number==7)
    inputs=[];last=None
    for number,_,value in fields(graph):
        if number==5:last=value
        elif number==11:
            inputs.append(next(bytes(v).decode() for n,_,v in fields(value) if n==1))
    if last is None:raise ValueError('Missing initializer')
    dims=[];raw=None;datatype=None
    for number,wire,value in fields(last):
        if number==1:
            if wire==0:dims.append(value)
            else:
                cursor=0
                while cursor<len(value):
                    dimension,cursor=varint(value,cursor);dims.append(dimension)
        elif number==2:datatype=value
        elif number==9 or (number==4 and wire==2):raw=value
    if dims!=[512,512] or datatype!=1 or raw is None or len(raw)!=512*512*4:
        raise ValueError('Unexpected official embedding map layout')
    matrix=np.frombuffer(raw,dtype='<f4').reshape(512,512).copy()
    if not np.isfinite(matrix).all():raise ValueError('Nonfinite embedding map')
    return matrix,inputs


class OpenCVReference:
    def __init__(self,gpu=False):
        path=ASSETS/'inswapper_128.onnx'
        with path.open('rb') as file:
            if hashlib.file_digest(file,'sha256').hexdigest()!=SWAP_SHA:raise ValueError('Official swap model digest mismatch')
        with (ASSETS/'w600k_r50.onnx').open('rb') as file:
            if hashlib.file_digest(file,'sha256').hexdigest()!=RECOGNITION_SHA:raise ValueError('Recognition model digest mismatch')
        self.emap,self.inputs=read_embedding_map(path)
        self.recognition=cv2.dnn.readNetFromONNX(str(ASSETS/'w600k_r50.onnx'))
        self.gpu=gpu
        if gpu:
            from ml.experiments.reference_backend.torch_reference import TorchResearchSwapper
            self.swapper=TorchResearchSwapper()
        else:self.swapper=cv2.dnn.readNetFromONNX(str(path))
        self.recognition.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        if not gpu:self.swapper.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        cv2.setNumThreads(4)

    def warmup(self):
        # Pay convolution setup at startup, not inside a timed browser frame.
        blank=np.zeros((112,112,3),dtype=np.uint8)
        latent=self.source_identity(blank,CANONICAL)
        if self.gpu:
            self.swapper(np.zeros((1,3,128,128),dtype=np.float32),latent).cpu().numpy()
        else:
            self.swapper.setInput(np.zeros((1,3,128,128),dtype=np.float32),self.inputs[0])
            self.swapper.setInput(latent,self.inputs[1]);self.swapper.forward()

    def source_identity(self,rgb,keypoints):
        affine=estimate_umeyama_similarity_transform(keypoints,CANONICAL)
        crop=cv2.warpAffine(np.asarray(rgb),affine,(112,112))
        blob=(crop.astype(np.float32).transpose(2,0,1)[None]-127.5)/127.5
        self.recognition.setInput(blob)
        embedding=self.recognition.forward().reshape(1,512)
        embedding/=max(np.linalg.norm(embedding),1e-8)
        latent=embedding@self.emap
        return latent/max(np.linalg.norm(latent),1e-8)

    def swap(self,rgb,keypoints,latent):
        canonical=CANONICAL.copy();canonical[:,0]+=8
        affine=estimate_umeyama_similarity_transform(keypoints,canonical)
        crop=cv2.warpAffine(np.asarray(rgb),affine,(128,128))
        blob=crop.astype(np.float32).transpose(2,0,1)[None]/255
        if len(self.inputs)!=2:raise ValueError('Unexpected swap input count')
        if self.gpu:result=self.swapper(blob,latent).cpu().numpy()
        else:
            self.swapper.setInput(blob,self.inputs[0]);self.swapper.setInput(latent,self.inputs[1])
            result=self.swapper.forward()
        if result.shape!=(1,3,128,128) or not np.isfinite(result).all():raise ValueError('Invalid reference output')
        return np.clip(result[0].transpose(1,2,0)*255,0,255).astype(np.uint8),affine
