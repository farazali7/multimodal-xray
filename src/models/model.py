import torch.nn as nn

from src.models.transformer import ViTEncoder
from src.models.image_encoders import get_biovil_image_encoder
from src.models.text_encoders import get_cxr_bert_tokenizer_and_encoder, get_text_embeddings
from src.models.decoder import Decoder


class ModelV1(nn.Module):
    def __init__(self, vit_args: dict, decoder_args: dict):
        """Final model v1.0

            Args:
                vit_args: Dictionary of kwargs for ViT encoder
                decoder_args: Dictionary of kwargs for decoder
        """
        super(ModelV1, self).__init__()

        # Image encoder
        self.image_encoder = get_biovil_image_encoder()

        # Text encoder
        self.text_tokenizer, self.text_encoder = get_cxr_bert_tokenizer_and_encoder()

        # ViT encoder
        self.vit_encoder = ViTEncoder(**vit_args)

        # Decoder
        self.decoder = Decoder(**decoder_args)

    def forward(self, x_img, x_txt):
        # Assume x_img is of shape [B, H, W] and x_txt is of shape [B, ?, ?]

        # Encode image
        image_embeddings = self.image_encoder(x_img)

        # TODO: Add positional embeddings to the image embeddings?

        # Get output from ViT encoder
        vit_embeddings = self.vit_encoder(image_embeddings)

        # Tokenize and encode report
        report_embeddings = get_text_embeddings(x_txt, self.text_tokenizer, self.text_encoder)

        # Decoder w/ cross-attention
        out = self.decoder(s1=vit_embeddings, s2=report_embeddings)

        return out
