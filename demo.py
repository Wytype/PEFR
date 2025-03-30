from model.backbones.body import Body
from utils import util
import os.path as osp
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def save_results(filename, canvas, candidates, subsets, save_dir):
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_dir = osp.join(save_dir, 'img')
    candidates_dir = osp.join(save_dir, 'candidates')
    subset_dir = osp.join(save_dir, 'subsets')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(candidates_dir):
        os.makedirs(candidates_dir)
    if not os.path.exists(subset_dir):
        os.makedirs(subset_dir)
    # 保存处理后的图像
    cv2.imwrite(osp.join(img_dir, filename), canvas)
    # 保存candidate和subset数据
    np.save(osp.join(candidates_dir, f'{os.path.splitext(filename)[0]}.npy'), candidates)
    np.save(osp.join(subset_dir, f'{os.path.splitext(filename)[0]}.npy'), subsets)



def save_results_person_main(filename, canvas, save_dir):
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 保存处理后的图像
    cv2.imwrite(osp.join(save_dir, filename), canvas)


body_estimation = Body('/data1/wuyue/reid/models/pretrained_checkpoints/body_pose_model.pth')
# root = '/data1/wuyue/reid/Occluded-DukeMTMC-Dataset/Occluded_Duke'
# train_dir = osp.join(root, 'bounding_box_train')
# query_dir = osp.join(root, 'query')
# gallery_dir = osp.join(root, 'bounding_box_test')
# save_dir = '/data1/wuyue/reid/Occluded-DukeMTMC-Dataset/Occluded_Duke_body_xmiddle'
# body_train_dir = osp.join(save_dir, 'bounding_box_train')
# body_query_dir = osp.join(save_dir, 'query')
# body_gallery_dir = osp.join(save_dir, 'bounding_box_test')

#Market1501
# root = '/data1/wuyue/reid/Market-1501'
# train_dir = osp.join(root, 'bounding_box_train')
# query_dir = osp.join(root, 'query')
# gallery_dir = osp.join(root, 'bounding_box_test')
# save_dir = '/data1/wuyue/reid/Market-1501/Occluded_Duke_body_xmiddle'
# body_train_dir = osp.join(save_dir, 'bounding_box_train')
# body_query_dir = osp.join(save_dir, 'query')
# body_gallery_dir = osp.join(save_dir, 'bounding_box_test')

#dukemtmc
root = '/data1/wuyue/reid/DukeMTMC-reID'
train_dir = osp.join(root, 'bounding_box_train')
query_dir = osp.join(root, 'query')
gallery_dir = osp.join(root, 'bounding_box_test')
save_dir = '/data1/wuyue/reid/DukeMTMC-reID/DukeMTMC_body_xmiddle'
body_train_dir = osp.join(save_dir, 'bounding_box_train')
body_query_dir = osp.join(save_dir, 'query')
body_gallery_dir = osp.join(save_dir, 'bounding_box_test')

# # Partial
# root = '/data1/wuyue/reid/Partial_REID'
# query_dir = osp.join(root, 'partial_body_images')
# gallery_dir = osp.join(root, 'whole_body_images')
# save_dir = '/data1/wuyue/reid/Partial_REID_xmiddle'
# body_query_dir = osp.join(save_dir, 'partial_body_images')
# body_gallery_dir = osp.join(save_dir, 'whole_body_images')

# # Occ_reid
# root = '/data1/wuyue/reid/Occ/Occluded_REID'
# query_dir = osp.join(root, 'occluded_body_images')
# gallery_dir = osp.join(root, 'whole_body_images')
# save_dir = '/data1/wuyue/reid/Occ/Occluded_REID_xmiddle'
# body_query_dir = osp.join(save_dir, 'occluded_body_images')
# body_gallery_dir = osp.join(save_dir, 'whole_body_images')

# 确保保存目录存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# if not os.path.exists(body_train_dir):
#     os.makedirs(body_train_dir)
if not os.path.exists(body_query_dir):
    os.makedirs(body_query_dir)
if not os.path.exists(body_gallery_dir):
    os.makedirs(body_gallery_dir)

image_paths = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))]
# 处理训练集图像
for image_path in tqdm(image_paths, desc='Processing images'):
    oriImg = cv2.imread(osp.join(train_dir, image_path))
    candidate, subset = body_estimation(oriImg)
    people = len(subset)
    if people <= 1:
        canvas = util.draw_bodypose(oriImg, candidate, subset)
    else:
        x_mean = np.zeros(len(subset))
        y_mean = np.zeros(len(subset))
        # 初始化计数器
        key_count = np.zeros(len(subset))
        # 遍历每个图像的 subset
        for n in range(len(subset)):
            for i in range(18):
                index = int(subset[n][i])
                if index != -1:
                    x_mean[n] += candidate[index, 0]
                    y_mean[n] += candidate[index, 1]
                    key_count[n] += 1
        # 计算平均值，避免除以零
        x_mean = np.where(key_count > 0, x_mean / key_count, 0)
        y_mean = np.where(key_count > 0, y_mean / key_count, 0)
        x = oriImg.shape[1]
        x_center = x / 2
        # 计算每个人与图像中心的水平距离
        dis = np.abs(x_mean - x_center)
        # 如果 dis[i] > x/4，将 dis[i] 设为 255
        dis[dis > x / 4.3] = 255
        flag = 1
        if len(dis) % 2 == 0:
            closest_two_indices = np.argsort(dis)[0:2]  # 取最小两个值的index,若最小值里有255那么不取
            if dis[closest_two_indices[0]] == 255:
                flag = 0
            elif dis[closest_two_indices[1]] == 255:
                closest_one_index = closest_two_indices[0]
            else:
                # 在最小的两个索引的人中找y小的，即在图像中上面的人
                if y_mean[closest_two_indices[0]] < y_mean[closest_two_indices[1]]:
                    closest_one_index = closest_two_indices[0]
                else:
                    closest_one_index = closest_two_indices[1]
        else:
            closest_one_index = np.argsort(dis)[0]  # 取最小一个值的index

        if flag:
            # 获取 subset 里第 closest_one_index 索引的数据，并将其转换为 [1, i] 形状的数组
            main_subset = np.expand_dims(subset[closest_one_index], axis=0)
        else:
            main_subset = np.empty((0, 20))  # [0,20]numpy
        canvas = util.draw_bodypose(oriImg, candidate,main_subset)

    save_results_person_main(image_path, canvas, body_train_dir)

image2_paths = [f for f in os.listdir(query_dir) if f.endswith(('.jpg', '.png', 'tif'))]
# 处理训练集图像
for image2_path in tqdm(image2_paths, desc='Processing images'):
    oriImg = cv2.imread(osp.join(query_dir, image2_path))
    candidate, subset = body_estimation(oriImg)
    people = len(subset)
    if people <= 1:
        canvas = util.draw_bodypose(oriImg, candidate, subset)
    else:
        x_mean = np.zeros(len(subset))
        y_mean = np.zeros(len(subset))
        # 初始化计数器
        key_count = np.zeros(len(subset))
        # 遍历每个图像的 subset
        for n in range(len(subset)):
            for i in range(18):
                index = int(subset[n][i])
                if index != -1:
                    x_mean[n] += candidate[index, 0]
                    y_mean[n] += candidate[index, 1]
                    key_count[n] += 1
        # 计算平均值，避免除以零
        x_mean = np.where(key_count > 0, x_mean / key_count, 0)
        y_mean = np.where(key_count > 0, y_mean / key_count, 0)
        x = oriImg.shape[1]
        x_center = x / 2
        # 计算每个人与图像中心的水平距离
        dis = np.abs(x_mean - x_center)
        # 如果 dis[i] > x/4，将 dis[i] 设为 255
        dis[dis > x / 4.3] = 255
        flag = 1
        if len(dis) % 2 == 0:
            closest_two_indices = np.argsort(dis)[0:2]  # 取最小两个值的index,若最小值里有255那么不取
            if dis[closest_two_indices[0]] == 255:
                flag = 0
            elif dis[closest_two_indices[1]] == 255:
                closest_one_index = closest_two_indices[0]
            else:
                # 在最小的两个索引的人中找y小的，即在图像中上面的人
                if y_mean[closest_two_indices[0]] < y_mean[closest_two_indices[1]]:
                    closest_one_index = closest_two_indices[0]
                else:
                    closest_one_index = closest_two_indices[1]
        else:
            closest_one_index = np.argsort(dis)[0]  # 取最小一个值的index

        if flag:
            # 获取 subset 里第 closest_one_index 索引的数据，并将其转换为 [1, i] 形状的数组
            main_subset = np.expand_dims(subset[closest_one_index], axis=0)
        else:
            main_subset = np.empty((0, 20))  # [0,20]numpy
        canvas = util.draw_bodypose(oriImg, candidate, main_subset)

    save_results_person_main(image2_path, canvas, body_query_dir)

image3_paths = [f for f in os.listdir(gallery_dir) if f.endswith(('.jpg', '.png', 'tif'))]
# 处理训练集图像
for image_path in tqdm(image3_paths, desc='Processing images'):
    oriImg = cv2.imread(osp.join(gallery_dir, image_path))
    candidate, subset = body_estimation(oriImg)
    people = len(subset)
    if people <= 1:
        canvas = util.draw_bodypose(oriImg, candidate, subset)
    else:
        x_mean = np.zeros(len(subset))
        y_mean = np.zeros(len(subset))
        # 初始化计数器
        key_count = np.zeros(len(subset))
        # 遍历每个图像的 subset
        for n in range(len(subset)):
            for i in range(18):
                index = int(subset[n][i])
                if index != -1:
                    x_mean[n] += candidate[index, 0]
                    y_mean[n] += candidate[index, 1]
                    key_count[n] += 1
        # 计算平均值，避免除以零
        x_mean = np.where(key_count > 0, x_mean / key_count, 0)
        y_mean = np.where(key_count > 0, y_mean / key_count, 0)
        x = oriImg.shape[1]
        x_center = x / 2
        # 计算每个人与图像中心的水平距离
        dis = np.abs(x_mean - x_center)
        # 如果 dis[i] > x/4，将 dis[i] 设为 255
        dis[dis > x / 4.3] = 255
        flag = 1
        if len(dis) % 2 == 0:
            closest_two_indices = np.argsort(dis)[0:2]  # 取最小两个值的index,若最小值里有255那么不取
            if dis[closest_two_indices[0]] == 255:
                flag = 0
            elif dis[closest_two_indices[1]] == 255:
                closest_one_index = closest_two_indices[0]
            else:
                # 在最小的两个索引的人中找y小的，即在图像中上面的人
                if y_mean[closest_two_indices[0]] < y_mean[closest_two_indices[1]]:
                    closest_one_index = closest_two_indices[0]
                else:
                    closest_one_index = closest_two_indices[1]
        else:
            closest_one_index = np.argsort(dis)[0]  # 取最小一个值的index

        if flag:
            # 获取 subset 里第 closest_one_index 索引的数据，并将其转换为 [1, i] 形状的数组
            main_subset = np.expand_dims(subset[closest_one_index], axis=0)
        else:
            main_subset = np.empty((0, 20))  # [0,20]numpy
        canvas = util.draw_bodypose(oriImg, candidate, main_subset)

    save_results_person_main(image_path, canvas, body_gallery_dir)



# image_paths = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))]
# # 处理训练集图像
# for image_path in tqdm(image_paths, desc='Processing images'):
#     oriImg = cv2.imread(osp.join(train_dir, image_path))
#     candidate, subset = body_estimation(oriImg)
#     canvas = util.draw_bodypose(oriImg, candidate, subset)
#     save_results(image_path, canvas, candidate, subset, body_train_dir)
#
# image2_paths = [f for f in os.listdir(query_dir) if f.endswith(('.jpg', '.png'))]
# # 处理训练集图像
# for image2_path in tqdm(image2_paths, desc='Processing images'):
#     oriImg = cv2.imread(osp.join(query_dir, image2_path))
#     candidate, subset = body_estimation(oriImg)
#     canvas = util.draw_bodypose(oriImg, candidate, subset)
#     save_results(image2_path, canvas, candidate, subset, body_query_dir)
#
# image3_paths = [f for f in os.listdir(gallery_dir) if f.endswith(('.jpg', '.png'))]
# # 处理训练集图像
# for image_path in tqdm(image3_paths, desc='Processing images'):
#     oriImg = cv2.imread(osp.join(gallery_dir, image_path))
#     candidate, subset = body_estimation(oriImg)
#     canvas = util.draw_bodypose(oriImg, candidate, subset)
#     save_results(image_path, canvas, candidate, subset, body_gallery_dir)