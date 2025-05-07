import numpy as np
import h5py  # 用于处理和存储 h5 格式文件
import os  # 用于文件和目录操作
import random  # 用于生成随机数
import numpy as np
import nibabel as nib  # 用于加载和处理医学影像文件 (NIfTI 格式)
import torch  # 用于深度学习框架 PyTorch 的处理

# 判断是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取影像数据，并返回归一化后的影像数据和全零标签
def read_data(data_path_img):
    # 拼接图像的完整路径
    src_data_file = os.path.join(data_path_img)

    # 使用 nibabel 加载 NIfTI 图像文件
    src_data_vol = nib.load(src_data_file)

    # 从加载的文件中获取影像数据
    images = src_data_vol.get_fdata()

    # 获取图像的体素尺寸（spacing），可以用于后续的图像处理
    spacing = src_data_vol.header['pixdim'][1:4]

    # 获取图像的宽 (w)，高 (h) 和深度 (d)
    w, h, d = images.shape

    # 输出图像的尺寸信息（归一化之前）
    print("images size (before norm): ", images.shape)

    # 将图像数据转换为 PyTorch 张量，并确保数据类型为浮点数
    images = torch.from_numpy(images.astype(float))

    # 对影像数据进行裁剪，将所有小于 500 的值设置为 500，大于 2500 的值设置为 2500
    images[images < 500] = 500
    images[images > 2500] = 2500

    # 输出图像的尺寸信息（归一化过程中）
    print("images size (in norm): ", images.shape)

    # 对影像数据进行归一化处理，范围变为 [0, 1]
    images = (images - 500) / (2500 - 500)

    # 将影像数据转移到 GPU 或 CPU 上
    images = images.to(device)  # 将数据转移到 GPU 或 CPU

    # 如果数据在 GPU 上，则转回 CPU，然后转换为 NumPy 数组
    images = images.cpu().numpy()  # 转换回 NumPy 数组

    # 输出图像的尺寸信息（归一化之后）
    print("images size (after norm): ", images.shape)

    # 返回全零标签，尺寸与影像数据一致
    labels = np.zeros((w, h, d))

    return images, labels  # 返回影像和标签数据


# 对影像和标签进行随机裁剪
def random_crop(image, label):
    output_size = (112, 112, 80)

    # 如果标签的尺寸小于裁剪输出尺寸，则进行填充
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= output_size[2]:
        # 计算需要填充的宽度、高度和深度
        pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
        pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)

        # 使用零填充（constant mode），对图像和标签进行填充
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

    # 获取填充后的影像的宽、高和深度
    (w, h, d) = image.shape

    # 初始化裁剪后的图像和标签列表
    image_list, label_list = [], []

    # 进行 15 次随机裁剪
    while len(image_list) < 15:
        # 随机生成裁剪的起始坐标 (w1, h1, d1)
        w1 = np.random.randint(0, w - output_size[0])
        h1 = np.random.randint(0, h - output_size[1])
        d1 = np.random.randint(0, d - output_size[2])

        # 从影像和标签中裁剪出指定大小的区域
        label_patch = label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
        image_patch = image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]

        # 将裁剪出的图像和标签加入到列表中
        label_list.append(label_patch)
        image_list.append(image_patch)

    return image_list, label_list  # 返回裁剪后的图像和标签


# 将裁剪后的图像和标签保存为 h5 格式
def covert_h5(images, labels, data_id):
    # 进行随机裁剪，获取裁剪后的图像和标签
    image_list, label_list = random_crop(images, labels)

    # 对每个裁剪出的图像块，生成对应的 h5 文件并保存
    for file_i in range(len(image_list)):
        # 指定保存路径，包括文件名，数据编号和裁剪块编号

        save_path = r"F:\dataset\CBCT_40_160\CBCT_data\unlabeled_" + str(data_id) + '_' + str(file_i)

        # 如果保存路径不存在，则创建路径
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 创建一个新的 h5 文件，保存裁剪出的图像和标签
        f = h5py.File(os.path.join(save_path, 'CBCT_roi.h5'), 'w')
        f.create_dataset('image', data=image_list[file_i])
        f.create_dataset('label', data=label_list[file_i])
        f.close()


# 主函数，处理图像列表中的数据
if __name__ == '__main__':

    # 打开未标注数据的图像路径列表文件
    with open(r"F:\dataset\40_160_unlabel_images.list", 'r') as f:
        image_list = f.readlines()

    # 去除每个路径字符串中的换行符
    image_list = [item.replace('\n', '') for item in image_list]

    # 遍历图像列表，逐个处理图像
    for data_id in range(len(image_list)):

        # 提取当前文件的原始文件名
        original_file_name = os.path.basename(image_list[data_id])  # 提取文件名，如 "11.nii.gz"
        file_number = original_file_name.split('.')[0]  # 提取编号部分，例如 "11"

        print('process the data:', file_number)

        # 读取影像数据，返回影像和全零标签
        images, labels = read_data(image_list[data_id])  # labels使用全0数组
        # 将影像和标签转换为 h5 文件，随机裁剪并保存
        covert_h5(images, labels, file_number)   # 转换Ctooth无标注数据至h5，随机裁剪