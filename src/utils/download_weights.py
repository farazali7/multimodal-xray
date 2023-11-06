import tempfile
from pathlib import Path

from torchvision.datasets.utils import download_url


def download_pretrained_weights(weights_url: str, filename: str, md5_hash: str) -> Path:
    """Download pretrained weights of a model and retrieve path to its checkpoint.

    Args:
        weights_url: URL of weights file
        filename: Name of file
        md5_hash: MD5 hash of weights file

    Returns:
        Path to weights checkpoint file on local machine.
    """

    root_dir = tempfile.gettempdir()
    download_url(weights_url, root=root_dir, filename=filename, md5=md5_hash)

    return Path(root_dir, filename)