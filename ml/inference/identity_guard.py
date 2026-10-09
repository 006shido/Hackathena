"""Reject swaps that lose the source to the target under a frozen independent scorer."""
import numpy as np
import torch
import torch.nn.functional as F
from ml.training.face_preprocessing import align_face_similarity,normalize_image_tensor


class IdentityGuard:
    def __init__(self,model,source,landmarks):
        self.model=model
        self.device=next(model.parameters()).device
        aligned,_,_=align_face_similarity(source,landmarks,target_size=128)
        with torch.inference_mode():self.source=model(self.tensor(aligned))

    def tensor(self,image):
        return torch.from_numpy(normalize_image_tensor(image)).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def check(self,target,output):
        embeddings=self.model(torch.cat([self.tensor(target),self.tensor(output)],dim=0))
        source_similarity=float(F.cosine_similarity(embeddings[1:2],self.source))
        target_similarity=float(F.cosine_similarity(embeddings[1:2],embeddings[:1]))
        accepted=bool(np.isfinite(source_similarity) and np.isfinite(target_similarity)
            and source_similarity>=.30 and source_similarity-target_similarity>=.05)
        return dict(identity_accepted=accepted,source_similarity=source_similarity,target_similarity=target_similarity)
