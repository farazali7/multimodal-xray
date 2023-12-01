from src.models.text_encoders import get_text_embeddings, get_cxr_bert_tokenizer_and_encoder
from src.models.model import ModelV2
from src.models.image_encoders import VQGanVAE

import torch
from torchvision.utils import save_image
import os

from typing import List

from config import cfg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_synthetic_cxr(model, vae, txt_tok, txt_enc, prompt: List[str] = None, temperature: float = 1.0,
                           timesteps: int = 18, topk_filter_thresh=0.9, image_name: str = "image"):
    """

    Args:
        model: Transformer model
        vae: VQGAN VAE model
        txt_tok: Text tokenizer model
        txt_enc: Encoder model
        prompt: Input prompt embeddings

    Returns:
        The decoded CXR message
    """
    prompt_emb = get_text_embeddings(prompt, txt_tok, txt_enc, device=device)

    synthetic = model.generate(prompt_emb, vae, temperature=temperature,
                               timesteps=timesteps, topk_filter_thresh=topk_filter_thresh)

    save_image(synthetic[0], os.path.join('results/images', image_name) + ".png")

    return synthetic


if __name__ == "__main__":
    model_args = cfg['MODEL']['ModelV2']
    encoder_args = model_args['ENCODER']
    projector_args = model_args['PROJECTOR']
    decoder_args = model_args['DECODER']

    txt_tok, txt_enc = get_cxr_bert_tokenizer_and_encoder()

    vae = VQGanVAE(**encoder_args, device=device)

    vae = vae.to(device)

    prompt = ['Pneumothorax']

    model = ModelV2(txt_enc, decoder_args, projector_args, device=device)

    ckpt_path = 'results/perc005_1eminus4_nomaskloss/epoch=21-step=836.ckpt'
    checkpoint = torch.load(ckpt_path, map_location=device)
    sd = {x.replace('model.', '') : v for x, v in checkpoint['state_dict'].items()}
    model.load_state_dict(sd)

    model = model.to(device)

    print(f"GENERATING WITH CKPT PATH: {ckpt_path}...")
    # Create and save pictures
    cxr1 = generate_synthetic_cxr(model, vae, txt_tok, txt_enc, prompt, 1.0, 18, 0.9, 'image1')
    cxr2 = generate_synthetic_cxr(model, vae, txt_tok, txt_enc, prompt, 1.0, 10, 0.9, 'image2')

    print('Done')
