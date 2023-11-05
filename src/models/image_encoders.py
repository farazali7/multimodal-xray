from src.utils.download_weights import download_pretrained_weights
from src.models.resnet import ResNet50Extractor

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
