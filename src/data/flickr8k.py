import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from PIL import Image
from PIL.Image import fromarray


class Flickr8kDataset(Dataset):
    """Handle dataset Flickr 8K, merge
        - dataset.py
        - preprocessing.py
    """
    def __init__(path_image, path_captions, vocab, transform, split):
        pass

    def __getitem__() -> (image_tensor, caption_tensor):
        pass
        