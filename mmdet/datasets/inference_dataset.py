import os

import numpy as np
from PIL import Image 

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import center_crop

# from torch.utils.data.dataset import T_co
from mmdet.datasets.pipelines import Compose

from mmcv.parallel import collate, scatter


def _pillow2array(img, flag='color', channel_order='bgr'):
    """Convert a pillow image to numpy array.

    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.

    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'unchanged':
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != 'RGB':
            if img.mode != 'LA':
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert('RGB')
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert('RGBA')
                img = Image.new('RGB', img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag == 'color':
            array = np.array(img)
            if channel_order != 'rgb':
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag == 'grayscale':
            img = img.convert('L')
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale" or "unchanged", '
                f'but got {flag}')
    return array


class batch_infer_dataset(Dataset):
    def __init__(self, root_dir, cfg, center_crop_ratio=0.8, img_list=None):
        self.root_dir = root_dir
        self.cfg = cfg
        self.cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
        self.twice = False
        if img_list is None:
            self.img_list = self._get_img_list()
        elif isinstance(img_list, list):
            self.img_list = img_list
            self.twice = True
        self.cfg_tf_pipeline = Compose(self.cfg.data.test.pipeline)

        self.STANDARD_HEIGHT = 720
        self.STANDARD_WIDTH = 1280
        self.CENTER_CROP_RATIO = center_crop_ratio

    def _get_img_list(self):
        all_list = os.listdir(self.root_dir)
        img_list = [f for f in all_list if f.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG'))]
        return img_list

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.img_list[idx])
        img = Image.open(img_name)

        if img.width < img.height:
            img = img.transpose(Image.ROTATE_90)
        if self.twice or (img.width > self.STANDARD_WIDTH and img.height > self.STANDARD_HEIGHT):
            cropHeight = int(img.height * self.CENTER_CROP_RATIO)
            cropWidth = int(img.width * self.CENTER_CROP_RATIO)
            img = center_crop(img, (cropHeight, cropWidth))
        img = _pillow2array(img)

        data = dict(img=img)
        data['ori_filename'] = img_name
        data = self.cfg_tf_pipeline(data)

        return data




