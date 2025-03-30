
import os.path as osp
import glob
from .bases import BaseImageDataset
import re
class Occluded_REID(BaseImageDataset):
    dataset_dir = 'Occluded_REID'
    dataset_train_dir = 'Market-1501'
    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(Occluded_REID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.dataset_train_dir = '/data1/wuyue/reid/Market-1501'
        self.dataset_body_train_dir = osp.join(self.dataset_train_dir, 'Occluded_Duke_body_xmiddle')
        self.train_dir = osp.join(self.dataset_train_dir, 'bounding_box_train')
        self.body_train_dir = osp.join(self.dataset_body_train_dir, 'bounding_box_train')
        # 增加body图像
        self.body_dir = osp.join(root, 'Occluded_REID_xmiddle')
        self.query_body_dir = osp.join(self.body_dir, 'occluded_body_images')
        self.gallery_body_dir = osp.join(self.body_dir, 'whole_body_images')

        self.query_dir = osp.join(self.dataset_dir, 'occluded_body_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'whole_body_images')

        self._check_before_run()
        self.pid_begin = pid_begin

        train = self._process_dir_train(self.train_dir, self.body_train_dir, relabel=True)
        query = self.process_dir(self.query_dir, self.query_body_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, self.gallery_body_dir, relabel=False, is_query=False)


        if verbose:
            print("=> Occluded_Reid loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def process_dir(self, dir_path, body_dir, relabel=False, is_query=True):
        img_paths = glob.glob(osp.join(dir_path,'*.tif'))
        if is_query:
            camid = 0
        else:
            camid = 1
        pid_container = set()
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            body_path = osp.join(body_dir, img_name)
            if relabel:
                pid = pid2label[pid]

            data.append((body_path, img_path, self.pid_begin +pid, camid,1))
        return data

    def _process_dir_train(self, dir_path, body_dir, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            filename = osp.basename(img_path)
            body_path = osp.join(body_dir, filename)
            dataset.append((body_path, img_path, self.pid_begin + pid, camid, 1))
        return dataset

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))


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


