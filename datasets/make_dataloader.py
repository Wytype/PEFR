import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import DataLoader

from .bases import ImageDataset
# from timm.data.random_erasing import RandomErasing
from utils.RandomE import RandomErasing
from .sampler import RandomIdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from .partial_reid import Paritial_REID
from .occlusion_reid import Occluded_REID


__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'occ_reid': Occluded_REID,
    'partial_reid': Paritial_REID,
}

#修改过
def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    bodys, imgs, pids, camids, viewids , _ = zip(*batch) ###
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(bodys, dim=0), torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    bodys, imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(bodys, dim=0), torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):

    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])




    # 定义变换序列
    train_transforms_A = A.Compose([
        A.Resize(width=cfg.INPUT.SIZE_TRAIN[0], height=cfg.INPUT.SIZE_TRAIN[1], interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=cfg.INPUT.PROB),
        A.PadIfNeeded(min_height=cfg.INPUT.SIZE_TRAIN[1] + 20, min_width=cfg.INPUT.SIZE_TRAIN[0] + 20, value=(0, 0, 0)),
        A.RandomCrop(width=cfg.INPUT.SIZE_TRAIN[0], height=cfg.INPUT.SIZE_TRAIN[1]),
        # ToTensorV2(),
        # A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ], additional_targets={'mask': 'mask'})

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms_A, val_transforms, change=True)
    train_set_normal = ImageDataset(dataset.train, train_transforms_A, val_transforms, change=False)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, train_transforms_A, val_transforms, change=False, train=False)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num



# class COCOSegmentationDataset(Dataset):
#     def __init__(self, img_dir, ann_file, num_classes=NUM_CLASSES, is_train=True):
#         super().__init__()
#         self.coco = COCO(ann_file)
#         self.img_dir = img_dir
#         self.num_classes = num_classes
#         self.is_train = is_train
#
#         # 数据增强：根据是否为训练集，选择不同的增强方式
#         if is_train:
#             self.transforms = A.Compose([
#                 A.Resize(384, 384, interpolation=1),  # 最近邻插值
#                 A.HorizontalFlip(p=0.5),
#                 A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=30, shift_limit=0.1, p=0.5, border_mode=0,
#                                    interpolation=1),  # 确保旋转和缩放时使用最近邻插值
#             ], additional_targets={'mask': 'mask'})
#         else:
#             self.transforms = A.Compose([
#                 A.Resize(384, 384, interpolation=1),  # 测试集同样使用最近邻插值
#             ], additional_targets={'mask': 'mask'})
#
#         # 归一化：无论是训练还是测试，图像都需要归一化
#         self.img_normalize = T.Compose([
#             T.ToTensor(),  # 转换为Tensor
#             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和方差
#         ])
#
#     def __len__(self):
#         return len(self.coco.imgs)
#
#     def __getitem__(self, idx):
#         img_id = self.coco.getImgIds()[idx]
#         img_info = self.coco.loadImgs(img_id)[0]
#         img_path = f"{self.img_dir}/{img_info['file_name']}"
#
#         # 打开图像
#         image = np.array(Image.open(img_path).convert("RGB"))
#
#         # 获取mask
#         ann_ids = self.coco.getAnnIds(imgIds=img_id)
#         anns = self.coco.loadAnns(ann_ids)
#         mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)  # 初始化mask
#
#         # 将mask叠加，并将类别id赋值到mask中
#         for ann in anns:
#             mask[self.coco.annToMask(ann) > 0] = ann['category_id']
#
#         # 应用几何变换（同步作用于image和mask）
#         transformed = self.transforms(image=image, mask=mask)
#
#         # 保证mask使用的是"最近邻插值"，避免生成非类别的值
#         image = transformed['image']
#         mask = transformed['mask']
#
#         # 对image进行归一化
#         image = self.img_normalize(Image.fromarray(image))
#
#         # 将mask转换为one-hot编码
#         mask = torch.from_numpy(mask).long()  # [H, W] 先转换为long类型
#         mask_onehot = torch.nn.functional.one_hot(mask, num_classes=self.num_classes)  # [H, W, num_classes]
#         mask_onehot = mask_onehot.permute(2, 0, 1).float()  # [num_classes, H, W]，转置维度
#         # unique_classes = torch.unique(mask).numpy()
#         # print(f"Unique classes in mask_tensor: {unique_classes}")
#         return image, mask_onehot