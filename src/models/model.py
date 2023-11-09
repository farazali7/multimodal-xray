import torch.nn as nn
import torch

from src.models.transformer import ViTEncoder
from src.models.image_encoders import get_biovil_image_encoder
from src.models.text_encoders import get_cxr_bert_tokenizer_and_encoder, get_text_embeddings
from src.models.decoder import Decoder
from src.models.attention import SinusoidalPositionalEmbeddings, LearnablePositionalEmbeddings


class ModelV1(nn.Module):
    def __init__(self, vit_args: dict, projector_args: dict, decoder_args: dict):
        """Final model v1.0

            Args:
                vit_args: Dictionary of kwargs for ViT encoder
                projector_args: Dictionary of kwargs for linear projector (matches dim of image & text embeddings)
                decoder_args: Dictionary of kwargs for decoder
        """
        super(ModelV1, self).__init__()

        # Image encoder
        self.image_encoder = get_biovil_image_encoder()

        # Image positional embeddings
        self.image_pos_emb = LearnablePositionalEmbeddings(embedding_dim=vit_args['embed_dim'])

        # Text encoder
        self.text_tokenizer, self.text_encoder = get_cxr_bert_tokenizer_and_encoder()

        # ViT encoder
        self.vit_encoder = ViTEncoder(**vit_args)

        # Image embeddings projector
        self.image_projector = nn.Linear(projector_args['img_embed_dim'], projector_args['out_dim'])
        self.text_projector = nn.Linear(projector_args['txt_embed_dim'], projector_args['out_dim'])

        # Decoder
        self.decoder = Decoder(**decoder_args)

    def forward(self, x_img, x_txt):
        # Assume x_img is of shape [B, H, W] and x_txt is of shape [B, ?, ?]

        # Encode image [B, H', W', Di], Di = 2048 right now without projection
        image_embeddings = self.image_encoder(x_img)

        # [B, T, Di]
        image_embeddings = torch.permute(torch.flatten(image_embeddings, 2, 3), (0, 2, 1))

        # Add positional embeddings to the image embeddings
        pos_embeddings = self.image_pos_emb(image_embeddings)
        image_embeddings = image_embeddings + pos_embeddings

        # Get output from ViT encoder
        # [B, T, Di]
        vit_embeddings = self.vit_encoder(image_embeddings)

        # Tokenize and encode report
        # [B, T, Dt], Dt = 128 right now (includes projection)
        report_embeddings = get_text_embeddings(x_txt, self.text_tokenizer, self.text_encoder,
                                                max_pad_len=vit_embeddings.shape[1])

        # Project image and text sequences to same dimensionality
        # Image & text embeddings shape [B, T, Dp], Dp = 128
        vit_embeddings = self.image_projector(vit_embeddings)
        report_embeddings = self.text_projector(report_embeddings)

        # Decoder w/ cross-attention
        out = self.decoder(s1=vit_embeddings, s2=report_embeddings)

        return out
