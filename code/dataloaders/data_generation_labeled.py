import numpy as np
import h5py
import os
import random
import numpy as np
import nibabel as nib
import torch

# 判断是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def free_memory():
    torch.cuda.empty_cache()  # 清理 GPU 缓存

def read_data(data_path_img, data_path_lab):
    src_data_file = os.path.join(data_path_img)
    src_data_vol = nib.load(src_data_file)
    images = src_data_vol.get_fdata()  # Use get_fdata() instead of get_data() for newer nibabel versions
    spacing = src_data_vol.header['pixdim'][1:4]  # 获取影像的体素尺寸（spacing）
    w, h, d = images.shape  # 获取影像的维度大小

    print("images size (before norm): ", images.shape)
    images = torch.from_numpy(images.astype(float))  # 将影像转换为 float 类型的 torch.Tensor
    images[images < 500] = 500  # 对影像进行阈值处理，设定最小值为500
    images[images > 2500] = 2500  # 设定最大值为2500
    print("images size (in norm): ", images.shape)
    images = (images - 500) / (2500 - 500)  # 对影像数据进行归一化处理

    # 将影像数据转移到 GPU 或 CPU 上
    images = images.to(device)  # 将数据转移到 GPU 或 CPU

    # 如果数据在 GPU 上，则转回 CPU，然后转换为 NumPy 数组
    images = images.cpu().numpy()  # 转换回 NumPy 数组
    print("images size (after norm): ", images.shape)

    # 在每次读取完数据后释放 GPU 内存
    free_memory()  # 清理 GPU 缓存

    # 加载标签数据
    lab_data_file = os.path.join(data_path_lab)
    lab_data_vol = nib.load(lab_data_file)
    labels = lab_data_vol.get_fdata()
    labels = torch.from_numpy(labels.astype(float))  # 转换为 torch.Tensor
    labels[labels > 0.5] = 1  # 阈值分割，label > 0.5 则设置为 1
    labels[labels <= 0.5] = 0  # label <= 0.5 则设置为 0

    # 将标签数据转移到 GPU 或 CPU 上
    labels = labels.to(device)  # 将数据转移到 GPU 或 CPU

    # 如果数据在 GPU 上，则转回 CPU，然后转换为 NumPy 数组
    labels = labels.cpu().numpy()  # 转换为 NumPy 数组

    # 在每次读取完数据后释放 GPU 内存
    free_memory()  # 清理 GPU 缓存

    return images, labels  # Return images and labels


def random_crop(image, label):
    output_size = (112, 112, 80)  # 定义输出尺寸

    # 如果数据尺寸比目标尺寸小，则对图像和标签进行 padding
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= output_size[2]:
        pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
        pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

    (w, h, d) = image.shape  # 获取影像维度
    image_list, label_list = [], []  # 存储裁剪后的图像和标签
    while len(image_list) < 15:  # 进行15次随机裁剪
        w1 = np.random.randint(0, w - output_size[0])
        h1 = np.random.randint(0, h - output_size[1])
        d1 = np.random.randint(0, d - output_size[2])

        # 从原始影像和标签中裁剪出小块
        label_patch = label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
        image_patch = image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]

        # 如果标签中有不同值并且目标区域像素数量足够多，保存这个 patch
        if len(np.unique(label_patch)) > 1 and np.sum(label_patch) >= 50000:
            print(np.sum(label_patch))
            label_list.append(label_patch)
            image_list.append(image_patch)

    return image_list, label_list  # 返回裁剪后的影像和标签


def covert_h5(images, labels, data_id):
    image_list, label_list = random_crop(images, labels)
    for file_i in range(len(image_list)):
        save_path = r"F:\dataset\CBCT_7_113\CBCT_data\labeled_" + str(data_id) + '_' + str(file_i)
        # save_path = r"F:\dataset\CBCT_7_113\CBCT_data\labeled_" + str(data_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        f = h5py.File(os.path.join(save_path, 'CBCT_roi.h5'), 'w')
        f.create_dataset('image', data=image_list[file_i])
        f.create_dataset('label', data=label_list[file_i])

        # f.create_dataset('image', data=images)
        # f.create_dataset('label', data=labels)

        f.close()


if __name__ == '__main__':

    with open(r"F:\dataset\CBCT_7_113\images.list", 'r') as f:  # CTooth标注 22例
        image_list = f.readlines()
    image_list = [item.replace('\n', '') for item in image_list]

    with open(r"F:\dataset\CBCT_7_113\labels.list", 'r') as f:
        label_list = f.readlines()
    label_list = [item.replace('\n', '') for item in label_list]

    for i in range(len(image_list)):

        # 提取当前文件的原始文件名
        original_file_name = os.path.basename(image_list[i])  # 提取文件名，如 "11.nii.gz"
        data_id = original_file_name.split('.')[0]  # 提取编号部分，例如 "11"

        print('process the data:', data_id)

        images, labels = read_data(image_list[i], label_list[i])

        covert_h5(images, labels, data_id)