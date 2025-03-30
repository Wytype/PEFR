import os
import torch
from PIL import Image
import os.path as osp
import glob
import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
# resize = A.Resize(width=224, height=224, interpolation=cv2.INTER_CUBIC)
# pad = A.pad(10)
# 定义变换序列
#0001_c2_f0046422.jpg
#0014_c5_f0058067.jpg
img = np.array(Image.open('/data1/wuyue/reid/Occluded-DukeMTMC-Dataset/Occluded_Duke/bounding_box_train/0014_c5_f0058067.jpg').convert('RGB'))
body = np.array(Image.open('/data1/wuyue/reid/Occluded-DukeMTMC-Dataset/Occluded_Duke_body_xmiddle/bounding_box_train/0014_c5_f0058067.jpg').convert('RGB'))

train_transforms_A = A.Compose([
    A.Resize(width=224, height=224, interpolation=3),
    A.HorizontalFlip(p=0.5),
    A.PadIfNeeded(min_height=224 + 20, min_width=224 + 20, value=(0, 0, 0)),
    A.RandomCrop(width=224, height=224),
], additional_targets={'mask': 'body'})

img_normalize = T.Compose([
            T.ToTensor(),  # 转换为Tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

img_toTensor = T.Compose([
    T.ToTensor()
])
#H,W,C
img2 = img_toTensor(img)  #C,H,W
body = img_toTensor(body)

mask = body-img2
mask = np.transpose(np.array(mask), (1, 2, 0))

single_channel_mask = np.sum(np.abs(mask), axis=2, keepdims=True)
# # print(single_channel_mask)
#
# #
# 步骤 2: 阈值化
# 将单通道图像转换为二值图像，阈值设置为 0.5（可以根据需要调整）
binary_mask = np.where(single_channel_mask >= 0.1, 1, 0)

# img = np.array(img)
# # #

#
transformed = train_transforms_A(image=img, mask=binary_mask)
img = transformed['image']
body = transformed['mask']

img = img_normalize(img)
body = img_toTensor(body)

mask_expanded = body.expand(3, -1, -1)  # 扩展到 (3, 224, 224)

ss = img*mask_expanded
ss = np.array(ss)
ss = np.transpose(ss, (1, 2, 0))
plt.imshow(ss[:, :, :])
plt.axis('off')
plt.show()
#
img = np.array(img)
body = np.array(body)
# print(img.shape)
# print(body.shape)
img = img.transpose((1,2,0))
body = body.transpose((1,2,0))
# plt.imshow(binary_mask[:, :, :])
# plt.axis('off')
# plt.show()
plt.imshow(img[:, :, :])
plt.axis('off')
plt.show()
plt.imshow(body[:, :, :])
plt.axis('off')
plt.show()
# train_dir = '/data1/wuyue/reid/Occluded-DukeMTMC-Dataset/Occluded_Duke/bounding_box_train'
# train2_dir = '/data1/wuyue/reid/Occluded-DukeMTMC-Dataset/Occluded_Duke_body_xmiddle/bounding_box_train'
# image_paths = glob.glob(osp.join(train_dir, '*.jpg'))
# image_paths2 = glob.glob(osp.join(train2_dir, '*.jpg'))
#
#
# image_names = [osp.basename(path) for path in image_paths]
# image_names2 = [osp.basename(path) for path in image_paths2]
# # image_paths = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))]
# # image_paths2 = [f for f in os.listdir(train2_dir) if f.endswith(('.jpg', '.png'))]
# if image_names == image_names2: print("两个文件夹的图片路径一致123")
# else: print("两个文件夹的图片路径不一致123")