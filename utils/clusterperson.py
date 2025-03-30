import torch
import torchvision.transforms as T
from datasets import make_dataloader
from model import make_model
import os
from model import make_model
from config import cfg
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="./configs/OCC_Duke/vit_oafr_224.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
if 'Market' in cfg.TEST.WEIGHT:
    model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
else:
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    # model = make_model(cfg, num_class=702, camera_num=camera_num, view_num=view_num)

model.load_param(cfg.TEST.WEIGHT)

single_person_refer = '/data1/reid/single_person'
multi_person_refer = '/data1/reid/multi_person'

device = "cuda"
model.to(device)
model.eval()
img_path_list = []
for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(train_loader_normal):
    with torch.no_grad():
        img = img.to(device)
        camids = camids.to(device)
        target_view = target_view.to(device)
        img_path_list.extend(imgpath)

        if cfg.MODEL.ARC == 'OAFR':
            globla_feats, local_feat_h, wei_h, co_h_pro = model(img, cfg.SOLVER.PERSON_OCC_PRO,
                                                                cam_label=camids,
                                                                view_label=target_view)
        else:
            feat = model(img, cam_label=camids, view_label=target_view)

