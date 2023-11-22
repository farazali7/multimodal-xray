from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt, log
import pickle
from typing import Optional

from src.utils.download_weights import download_pretrained_weights
from src.models.resnet import ResNet50Extractor
from src.utils.config_helpers import instantiate_from_cfg
from src.utils.taming_code.vqgan import GumbelVQ

# CONSTANTS
HF_URL = "https://huggingface.co"

BIOMED_VLP_CXR_BERT_SPECIALIZED = "microsoft/BiomedVLP-CXR-BERT-specialized"
CXR_BERT_COMMIT_TAG = "v1.1"

BIOVIL_IMAGE_WEIGHTS_NAME = "biovil_image_resnet50_proj_size_128.pt"
BIOVIL_IMAGE_WEIGHTS_URL = f"{HF_URL}/{BIOMED_VLP_CXR_BERT_SPECIALIZED}/resolve/{CXR_BERT_COMMIT_TAG}/{BIOVIL_IMAGE_WEIGHTS_NAME}"  # noqa: E501
BIOVIL_IMAGE_WEIGHTS_MD5 = "02ce6ee460f72efd599295f440dbb453"


def get_biovil_image_encoder() -> ResNet50Extractor:
    """ Download and return pretrained ResNet50 encoder from BioVil model.

    Returns:
        Pretrained ResNet50 image encoder.
    """
    ckpt = download_pretrained_weights(BIOVIL_IMAGE_WEIGHTS_URL, BIOVIL_IMAGE_WEIGHTS_NAME, BIOVIL_IMAGE_WEIGHTS_MD5)
    model = ResNet50Extractor(pretrained_weights=ckpt)

    return model


# NOTE: THE BELOW CLASSES ARE PURELY COPIED FROM
# https://github.com/CompVis/taming-transformers/blob/master/taming/models/vqgan.py
# SINCE THEY WERE UNAVAILABLE IN LATEST PIP INSTALL OF THAT GITHUB REPO
class VQGanVAE(nn.Module):
    def __init__(self, model_path: str, cfg_path: str, codebook_path: Optional[str] = None):
        super(VQGanVAE, self).__init__()
        config = OmegaConf.load(cfg_path)

        model = instantiate_from_cfg(config.model)

        state = torch.load(model_path, map_location='cpu')['state_dict']
        model.load_state_dict(state, strict=False)

        print(f"Loaded VQGAN from {model_path} and {cfg_path}")

        self.model = model

        self.f = config.model.params.ddconfig.resolution / config.model.params.ddconfig.attn_resolutions[0]
        self.fmap_size = config.model.params.ddconfig.attn_resolutions[0]
        self.num_layers = int(log(self.f) / log(2))
        self.image_size = config.model.params.ddconfig.resolution
        self.num_tokens = config.model.params.n_embed
        self.embed_dim = config.model.params.embed_dim
        self.is_gumbel = isinstance(self.model, GumbelVQ)

        with open(codebook_path, 'rb') as f:
            self.codebook_indices = pickle.load(f)
        print(f'Loaded codebook indices from {codebook_path}')

    @torch.no_grad()
    def get_codebook_indices_old(self, img):
        b = img.shape[0]
        # Rescale the image values so that they go from [0, 1] to [-1, 1]
        img = (2*img) - 1
        z_q, emb_loss, [perplexity, min_encodings, indices] = self.model.encode(img)
        indices = indices.squeeze(-1)
        if self.is_gumbel:
            return rearrange(indices, 'b h w -> b (h w)', b=b)
        return z_q, emb_loss, perplexity, rearrange(indices, '(b n) -> b n', b = b)

    @torch.no_grad()
    def get_codebook_indices(self, img_name):
        idx_list = self.codebook_indices[img_name]

        # Convert to tensor with batch size 1 (i.e., shape: (1, 1024))
        idx_tensor = torch.Tensor(idx_list)[None, ...].to(torch.int64)

        return idx_tensor

    def decode(self, img_seq):
        b, n = img_seq.shape

        one_hot_indices = F.one_hot(img_seq, num_classes=self.num_tokens).float()  # 1024
        z = one_hot_indices @ self.model.quantize.embed.weight if self.is_gumbel \
            else (one_hot_indices @ self.model.quantize.embedding.weight)
        z = rearrange(z, 'b (h w) c -> b c h w', h = int(sqrt(n)))

        img = self.model.decode(z)

        img = (img.clamp(-1., 1.) + 1) * 0.5

        return img

    def forward(self, img):
        raise NotImplemented
