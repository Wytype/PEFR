from PIL import Image, ImageFile
import torchvision.transforms as T
from torch.utils.data import Dataset
import albumentations as A
import os.path as osp
import cv2
import random
import torch
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.RandomE import RandomErasing
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, _, pid, camid, trackid in data:#zengjia
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform_A=None, transform=None, change=True, train=True):
        self.dataset = dataset
        self.transform_A = transform_A
        self.transform = transform
        self.resize_A = A.Compose([A.Resize(width=224,height=224,interpolation=cv2.INTER_CUBIC),], additional_targets={'mask': 'mask'})
        self.change = change
        self.train = train
        self.img_toTensor = T.Compose([
            T.ToTensor()
        ])
        self.img_normalize = T.Compose([
            T.ToTensor(),  # 转换为Tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.random_erasing = RandomErasing(probability=0.5, mode='pixel', max_count=1, device='cpu')
        self.kernel = np.ones((3, 3), np.uint8)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        body_path, img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        body = read_image(body_path)
        img = np.array(img)  # 转化为np，A.Compose无法处理PIL#H,W,C
        body = np.array(body)
        img2 = self.img_toTensor(img)  # C,H,W
        body = self.img_toTensor(body)
        mask = body - img2
        mask = np.transpose(np.array(mask), (1, 2, 0)) #H,W,C
        single_channel_mask = np.sum(np.abs(mask), axis=2, keepdims=True)
        binary_mask = np.where(single_channel_mask >= 0.1, 1, 0).astype(np.float32)
        if self.train:
            if np.sum(binary_mask) < 16:
                binary_mask = np.ones_like(binary_mask, dtype=np.float32)

        # 进行膨胀操作
        binary_mask = cv2.dilate(binary_mask, self.kernel, iterations=1)
        if self.transform_A is not None: #数据增强
            if self.change:
                transformed = self.transform_A(image=img, mask=binary_mask)
                img = transformed['image']
                body = transformed['mask']
                img = self.img_normalize(img)
                body = self.img_toTensor(body)
                img, body = self.random_erasing(img, body)
            else:
                transformed = self.resize_A(image=img, mask=binary_mask)
                img = transformed['image']
                body = transformed['mask']
                img = self.img_normalize(img)
                body = self.img_toTensor(body)

        return body, img, pid, camid, trackid,img_path.split('/')[-1]