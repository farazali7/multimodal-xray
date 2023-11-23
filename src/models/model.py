import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


from src.models.transformer import ViTEncoder, TransformerDecoder
from src.models.image_encoders import get_biovil_image_encoder, VQGanVAE
from src.models.text_encoders import get_cxr_bert_tokenizer_and_encoder, get_text_embeddings
from src.models.decoder import Decoder
from src.models.attention import SinusoidalPositionalEmbeddings, LearnablePositionalEmbeddings

import lightning as L


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

        # Encode image [B, H', W', Di]
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
        # [B, T, Dt]
        report_embeddings = get_text_embeddings(x_txt, self.text_tokenizer, self.text_encoder,
                                                max_pad_len=vit_embeddings.shape[1])

        # Project image and text sequences to same dimensionality
        # Image & text embeddings shape [B, T, Dp]
        vit_embeddings = self.image_projector(vit_embeddings)
        report_embeddings = self.text_projector(report_embeddings)

        # Decoder w/ cross-attention
        out = self.decoder(s1=vit_embeddings, s2=report_embeddings)

        return out


class ModelV2(nn.Module):
    def __init__(self, decoder_args: dict, projector_args: dict):
        """Final model v2.0 - uses masked vision token modelling

            Args:
                decoder_args: Dictionary of kwargs for Transformer decoder
                projector_args: Dictionary of kwargs for linear projector (matches dim of image & text embeddings)
        """
        super(ModelV2, self).__init__()

        self.image_embedding = nn.Embedding(num_embeddings=1024,
                                            embedding_dim=decoder_args['embed_dim'])

        # Image positional embeddings
        self.image_pos_emb = LearnablePositionalEmbeddings(embedding_dim=decoder_args['embed_dim'],
                                                           max_seq_len=1024)

        # Text embeddings projector (if dimensionality of text sequence is not same as transformer then project)
        self.text_projector = nn.Linear(projector_args['txt_embed_dim'], decoder_args['embed_dim']) if \
            projector_args['txt_embed_dim'] != decoder_args['embed_dim'] else nn.Identity()

        # Unmasking transformer
        self.transformer = TransformerDecoder(**decoder_args)

        # Final MLP
        self.final_dense = nn.Linear(in_features=decoder_args['embed_dim'], out_features=1024)

    def forward(self, x_img, x_txt):
        # Assume x_img is of shape [B, 1024] and x_txt is of shape [B, T, Dt]

        # TODO: MASK IMAGE INDICES

        # Transform indices to image embeddings [B, 1024, Dmodel]
        image_embeddings = self.image_embedding(x_img)

        # Add positional embeddings to the image embeddings
        pos_embeddings = self.image_pos_emb(image_embeddings)
        image_embeddings = image_embeddings + pos_embeddings

        # Project text sequences to same dimensionality as transformer
        report_embeddings = self.text_projector(x_txt)

        # Shapes before the decoder:
        # image_embeddings shape  - [B, 1024, Dmodel]
        # report_embeddings shape - [B, T, Dmodel]

        # Decoder w/ cross-attention
        out = self.transformer(x=image_embeddings, context=report_embeddings)

        logits = self.final_dense(out)

        return logits


class FinalModel(L.LightningModule):
    def __init__(self, model_def: str, model_args: dict, lr: float):
        """Lightning Module wrapper around a model.

        Args:
            model_def: String specifying name of model class
            model_args: kwargs to a model
            lr: Learning rate for model
        """
        super(FinalModel, self).__init__()
        self.model = eval(model_def)(model_args)
        self.lr = lr
        self.ssim = SSIM()
        self.fid = FID(feature=64)

    def training_step(self, batch, batch_idx):
        x_img, x_txt, y = batch
        out = self.model(x_img, x_txt)
        loss = F.mse_loss(out, y)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_img, x_txt, y = batch
        fids = []
        
        out = self.model(x_img, x_txt)
        val_loss = F.mse_loss(out, y)
        
        ssim = self.ssim(out,y)
        
        self.fid.update(y,real=True)
        self.fid.update(out,real=False)
        fids.append(self.fid.compute())

        self.log("val_loss", val_loss)
        self.log("SSIM",ssim)
        fig,ax = self.fid.plot(fids)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return optimizer, lr_scheduler
