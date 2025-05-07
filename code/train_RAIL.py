import os

import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from networks.vnet import VNet
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomRotFlip, ToTensor, TwoStreamBatchSampler

from test_util import test_all_case_array


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


from networks.ResVNet import ResVNet


import torch.nn as nn
import torch.nn.functional as F

import math

# 设置初始参数

parser = argparse.ArgumentParser()  # 创建命令行参数解析器


parser.add_argument(
    "--root_path",
    type=str,
    default="/hy-tmp/dataset/CTooth_13_109/CTooth_data",
    help="Name of Experiment",
)  # 指定数据集根目录的路径


parser.add_argument("--exp", type=str, default="RAIL", help="model_name")  # 指定模型名称


parser.add_argument(
    "--max_iterations", type=int, default=8000, help="maximum epoch number to train"
)  # 指定最大训练迭代次数


parser.add_argument(
    "--batch_size", type=int, default=2, help="batch_size per gpu"
)  # 指定每个GPU的批处理大小


parser.add_argument(
    "--labeled_bs", type=int, default=1, help="labeled_batch_size per gpu"
)  # 指定每个GPU的有标签样本批处理大小

parser.add_argument(
    "--base_lr", type=float, default=0.01, help="maximum epoch number to train"
)  # 指定初始学习率
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)  # 是否使用确定性训练
parser.add_argument("--seed", type=int, default=1337, help="random seed")  # 设置随机种子
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")  # 指定要使用的GPU编号
### costs
parser.add_argument(
    "--ema_decay", type=float, default=0.999, help="ema_decay"
)  # 指定指数移动平均的衰减率
parser.add_argument(
    "--consistency_type", type=str, default="mse", help="consistency_type"
)  # 指定一致性损失的类型
parser.add_argument(
    "--consistency", type=float, default=20.0, help="consistency"
)  # 指定一致性损失的权重
parser.add_argument(
    "--wd", type=float, default=0.0004, help="consistency"
) 
parser.add_argument(
    "--consistency_rampup", type=float, default=20.0, help="consistency_rampup"
)  # 指定一致性损失的斜坡上升时间
parser.add_argument("--model_num", type=int, default=2, help="model_num")

parser.add_argument("--labelnum", type=int, default=13, help="model_num")
parser.add_argument("--unlabelnum", type=int, default=107, help="model_num")
parser.add_argument("--warmup_iters", type=int, default=1000, help="warm up numbers")
parser.add_argument("--min_lr", type=float, default=1e-5, help="mini of learning rate")
parser.add_argument("--thred", type=float, default=0.5, help="change the lr stretagy")
parser.add_argument("--alfa", type=float, default=0.5, help="change the lr stretagy")
parser.add_argument("--beta", type=float, default=0.05, help="change the lr stretagy")

args = parser.parse_args()  # 解析命令行参数并存储在args变量中

# 设置模型数据路径
train_data_path = args.root_path
# epoch_step = args.epoch_step
# 设置训练完毕模型名称

snapshot_path = "../model/" + args.exp + "/"

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 设置可见的CUDA设备
batch_size = args.batch_size * len(args.gpu.split(","))  # 计算总的批处理大小
max_iterations = args.max_iterations  # 获取最大迭代次数
base_lr = args.base_lr  # 获取初始学习率
labeled_bs = args.labeled_bs  # 获取有标签样本的批处理大小
model_num = args.model_num
# model_step = epoch_step // model_num
model_is_first_term = [True] * model_num
ema_decay = args.ema_decay
# 检测是否进行确定性训练

best_score = 0


# 获取当前时间戳并格式化
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# 构造带时间戳的路径
log_dir = os.path.join(snapshot_path, f'tensorboard_logs', f"{timestamp}" )

# 创建 TensorBoard writer
writer = SummaryWriter(log_dir)


if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


num_classes = 2  # 指定分类任务的类别数
patch_size = (112, 112, 80)  # 指定图像的裁剪尺寸
T = 0.1  # sharpen function 参数
Good_student = 0  # 0: vnet 1:resnet 默认好学生(不影响)

def test_calculate_metric (model_array):

    
    with open(args.root_path + "/" + "../Flods/val.list", "r") as f:  # todo change test flod

    
        image_list = f.readlines()
    image_list = [
        
        
        args.root_path + "/" + item.replace("\n", "") + "/CBCT_roi.h5" for item in image_list
        
        
    ]
    
    test_save_path = "../model/prediction/" + args.exp  + "/" + "_post/"
    test_save_path = os.path.join(test_save_path, f"models_{timestamp}/")
    os.makedirs(test_save_path, exist_ok=True)
    avg_metric = test_all_case_array(
        model_array,
        image_list,
        num_classes=num_classes,
        patch_size=(112, 112, 80),
        stride_xy=64,
        stride_z=32,  
        save_result=True,
        test_save_path=test_save_path,
    )
    return avg_metric

# 一致性损失系数计算，通过一个sigmoid+exp实现warm up

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_teacher_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 10.0 * ramps.sigmoid_rampup(epoch, 20.0)


# 通过 EMA 更新参数


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


# 在多个工作进程中使用不同的随机种子

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)



def get_lr(iter_num, warmup_iters, thred, total_iters, base_lr, min_lr=1e-6):
    """
    分阶段学习率策略：
    - Warm-up: 线性增加到 base_lr
    - 恒定阶段: 维持 base_lr
    - 余弦衰减: 逐步衰减到 min_lr
    """
    if iter_num < warmup_iters:
        # 线性 warm-up
        lr = min_lr + (base_lr - min_lr) * (iter_num / warmup_iters)
    elif iter_num < total_iters * thred:
        # 维持恒定学习率
        lr = base_lr
    else:
        # 余弦衰减
        decay_iters = total_iters - total_iters * thred
        lr = min_lr + (base_lr - min_lr) * thred * (1 + math.cos(math.pi * (iter_num - total_iters * thred) / decay_iters))
    return lr


# 模型数据类
class ModelData:
    def __init__(self, outputs, label, labeled_bs, is_first_term=False):
        self.outputs = outputs
        self.loss_seg = F.cross_entropy(self.outputs[:labeled_bs], label[:labeled_bs])
        self.outputs_soft = F.softmax(self.outputs, dim=1)
        self.loss_seg_dice = losses.dice_loss(
            self.outputs_soft[:labeled_bs, 1, :, :, :], label[:labeled_bs] == 1
        )

        
        self.outputs_soft2 = F.softmax(self.outputs, dim=1)
        self.predict = torch.max(
            self.outputs_soft2[:labeled_bs, :, :, :, :],
            1,
        )[1]


        self.gt = torch.max(
            label[:labeled_bs, :, :, :],
            1,
        )[1]


        self.mse_dist = consistency_criterion(
            self.outputs_soft2[:labeled_bs, 1, :, :, :], label[:labeled_bs]
        )


        self.kl_dist = losses.softmax_kl_loss(
            self.outputs_soft2[:labeled_bs, 1, :, :, :], label[:labeled_bs]
        )


        self.outputs_clone = self.outputs_soft[labeled_bs:, :, :, :, :].clone().detach()
        self.outputs_clone1 = torch.pow(self.outputs_clone, 1 / T)
        self.outputs_clone2 = torch.sum(self.outputs_clone1, dim=1, keepdim=True)
        self.outputs_PLable = torch.div(self.outputs_clone1, self.outputs_clone2)
        self.is_first_term = is_first_term


        self.student_predict = torch.max(
            self.outputs_soft[labeled_bs:, :, :, :, :],
            1,
        )[1]


    def get_supervised_loss(self, diff_mask=None, err_mask=None):
        if diff_mask is None and err_mask is None:
            self.supervised_loss = self.loss_seg + self.loss_seg_dice
            self.loss = self.supervised_loss


        elif diff_mask is None:


            self.mse = self.mse_dist.mean()


            self.mse_new = torch.sum(err_mask * self.kl_dist) / (
                torch.sum(err_mask) + 1e-16
            )


            self.supervised_loss = (self.loss_seg + self.loss_seg_dice) + args.alfa * self.mse + args.beta * self.mse_new
            self.loss = self.supervised_loss
        else:
            self.mse = torch.sum(diff_mask * self.mse_dist) / (
                torch.sum(diff_mask) + 1e-16
            )


            self.mse_new = torch.sum(diff_mask * err_mask * self.kl_dist) / (
                torch.sum(diff_mask * err_mask) + 1e-16
            )
            
            
            self.supervised_loss = (self.loss_seg + self.loss_seg_dice) + args.alfa * self.mse + args.beta * self.mse_new
            
            
            self.loss = self.supervised_loss


    def add_teacher_loss(self, teacher_outputs, consistency_weight=0):
        self.teacher_outputs_soft = F.softmax(teacher_outputs, dim=1)


        self.teacher_predict = torch.max(
            self.teacher_outputs_soft[labeled_bs:, :, :, :, :],
            1,
        )[1]  ## 老师的预测


        self.teacher_outputs_clone = (
            self.teacher_outputs_soft[labeled_bs:, :, :, :, :].clone().detach()
        )
        self.teacher_outputs_clone1 = torch.pow(self.teacher_outputs_clone, 1 / T)
        self.teacher_outputs_clone2 = torch.sum(
            self.teacher_outputs_clone1, dim=1, keepdim=True
        )
        self.teacher_outputs_PLable = torch.div(
            self.teacher_outputs_clone1, self.teacher_outputs_clone2
        )
        
        self.teacher_dist = consistency_criterion(
            self.outputs_soft[labeled_bs:, :, :, :, :], self.teacher_outputs_PLable
        )
        b, v, w, h, d = self.teacher_dist.shape
        self.teacher_dist = torch.sum(self.teacher_dist) / (b * v * w * h * d)
        self.teacher_loss = self.teacher_dist


        self.loss = self.loss + consistency_weight * self.teacher_loss


    def get_loss(self, Plabel, consistency_weight=0, diff_mask=None):
        self.consistency_dist = consistency_criterion(
            self.outputs_soft[labeled_bs:, :, :, :, :], Plabel
        )
        b, c, w, h, d = self.consistency_dist.shape
        self.consistency_dist = torch.sum(self.consistency_dist) / (b * c * w * h * d)
        self.consistency_loss = self.consistency_dist


        if diff_mask is None:
            self.loss = self.supervised_loss + consistency_weight * self.consistency_loss
        else:
            voxel_kl_loss = nn.KLDivLoss(reduction="none")
            uniform_distri = torch.ones(self.outputs_soft[labeled_bs:, :, :, :, :].shape)
            uniform_distri = uniform_distri.cuda()

            self.consistency_loss_new = torch.mean(voxel_kl_loss(F.log_softmax(self.outputs_soft[labeled_bs:, :, :, :, :], dim=1), uniform_distri), dim=1)
            self.consistency_loss_new = torch.sum(diff_mask * self.consistency_loss_new) / (
                    torch.sum(diff_mask) + 1e-16
            )


            self.loss = self.supervised_loss + consistency_weight * self.consistency_loss + 0.1 * consistency_weight * self.consistency_loss_new


def train_model(
    model_array,
    teacher_model_array,
    optimizer_array,
    data_buffer,
    model_iter_num,
    idx,
    first_term=False,
):
    for sampled_batch in data_buffer:
        
        # 如果是第一次，初始化一下
        
        if first_term:
            
            # 计算有监督数据
            
            data_1 = ModelData(
                model_array[idx](sampled_batch[0]["image"].cuda()),
                sampled_batch[0]["label"].cuda(),
                labeled_bs,
                first_term,
            )


            data_2 = ModelData(
                model_array[idx + 2](sampled_batch[0]["image"].cuda()),
                sampled_batch[0]["label"].cuda(),
                labeled_bs,
                first_term,
            )

            
            # 计算损失权重
            
            consistency_weight = get_current_consistency_weight(
                model_iter_num[idx] // 150
            )
            teacher_weight = get_teacher_consistency_weight(model_iter_num[idx] // 150)
            
            # 计算有监督损失
            
            data_1.get_supervised_loss()


            data_2.get_supervised_loss()

          
            # 计算Teacher损失
            
            data_1.add_teacher_loss(
                teacher_model_array[idx](sampled_batch[0]["image"].cuda()),
                teacher_weight,
            )


            data_2.add_teacher_loss(
                teacher_model_array[idx](sampled_batch[0]["image"].cuda()),
                teacher_weight,
            )


            # 对于当前模型进行迭代
            
            optimizer_array[idx].zero_grad()
            data_1.loss.backward()
            optimizer_array[idx].step()


            optimizer_array[idx + 2].zero_grad()
            data_2.loss.backward()
            optimizer_array[idx + 2].step()

            
            # 迭代Teacher模型
            
            update_ema_variables(
                model_array[idx], teacher_model_array[idx], ema_decay, model_iter_num[idx]
            )


            # 记录当前模型的迭代次数
            
            model_iter_num[idx] += 1
        else:
            
            # 计算有监督数据
            
            data_arrays = []
            for i in range(model_num):
                data_arrays.append(
                    ModelData(
                        model_array[i](sampled_batch[0]["image"].cuda()),
                        sampled_batch[0]["label"].cuda(),
                        labeled_bs,
                    )
                )  ## 两个VNet的loss类


            data_arrays.append(
                ModelData(
                    model_array[2](sampled_batch[0]["image"].cuda()),
                    sampled_batch[0]["label"].cuda(),
                    labeled_bs,
                )
            )  


            data_arrays.append(
                ModelData(
                    model_array[3](sampled_batch[0]["image"].cuda()),
                    sampled_batch[0]["label"].cuda(),
                    labeled_bs,
                )
            )  ## 两个ResVNet的loss类


            # 使用 lambda 表达式和 min() 函数找到 loss_seg_dice 最小的 ModelData 对象
            min_loss_seg_dice_model = min(data_arrays, key=lambda x: x.loss_seg_dice)

            # 获取最小的 loss_seg_dice 值和对应的序号
            Good_student = data_arrays.index(min_loss_seg_dice_model)
            
            # 获得 Mask


            ## 同一组学生模型之间的标签预测diff_mask
            diff_mask = torch.logical_or(data_arrays[idx].predict == 1, data_arrays[idx + 2].predict == 1).to(torch.int32) - \
                        torch.logical_and(data_arrays[idx].predict == 1, data_arrays[idx + 2].predict == 1).to(torch.int32)

            ## VNet学生模型与Good_student模型之间的标签预测diff_mask_1
            diff_mask_1 = torch.logical_or(data_arrays[idx].predict == 1, data_arrays[Good_student].predict == 1).to(torch.int32) - \
                        torch.logical_and(data_arrays[idx].predict == 1, data_arrays[Good_student].predict == 1).to(torch.int32)

            ## ResVNet学生模型与Good_student模型之间的标签预测diff_mask_2
            diff_mask_2 = torch.logical_or(data_arrays[idx + 2].predict == 1, data_arrays[Good_student].predict == 1).to(torch.int32) - \
                        torch.logical_and(data_arrays[idx + 2].predict == 1, data_arrays[Good_student].predict == 1).to(torch.int32)



            err_mask_1 = torch.logical_or(data_arrays[idx].predict == 1, data_arrays[idx].gt == 1).to(torch.int32) - \
                        torch.logical_and(data_arrays[idx].predict == 1, data_arrays[idx].gt == 1).to(torch.int32)



            err_mask_2 = torch.logical_or(data_arrays[idx + 2].predict == 1, data_arrays[idx + 2].gt == 1).to(torch.int32) - \
                        torch.logical_and(data_arrays[idx + 2].predict == 1, data_arrays[idx + 2].gt == 1).to(torch.int32)


            if Good_student != idx and Good_student != (idx + 2):  ## 证明最好模型出现在上一轮训练的模型中

                data_arrays[idx].get_supervised_loss(diff_mask_1, err_mask_1)
                data_arrays[idx + 2].get_supervised_loss(diff_mask_2, err_mask_2)

            elif Good_student == idx:  ## 最好学生是本轮的VNet

                data_arrays[idx].get_supervised_loss(err_mask=err_mask_1)
                data_arrays[idx + 2].get_supervised_loss(diff_mask, err_mask_2)


            else:  ## 最好学生是本轮的ResVNet

                data_arrays[idx].get_supervised_loss(diff_mask, err_mask_1)
                data_arrays[idx + 2].get_supervised_loss(err_mask=err_mask_2)
                

            if idx == 0:
                data_arrays[idx + 1].get_supervised_loss()
                data_arrays[idx + 3].get_supervised_loss()

            elif idx == 1:
                data_arrays[idx - 1].get_supervised_loss()
                data_arrays[idx + 1].get_supervised_loss()
            

             # 记录每个模型的 supervised_loss 到 TensorBoard
            for i, data in enumerate(data_arrays):
                writer.add_scalar(f"Loss/Supervised_Model_{i}", data.supervised_loss.item(), model_iter_num[idx])
                writer.add_scalar(f"Loss/Segmentation_Model_{i}", data.loss_seg.item(), model_iter_num[idx])
                writer.add_scalar(f"Loss/Segmentation_Dice_Model_{i}", data.loss_seg_dice.item(), model_iter_num[idx])

            Plabel = data_arrays[Good_student].outputs_PLable

            # 计算损失权重
            consistency_weight = get_current_consistency_weight(
                model_iter_num[idx] // 150
            )
            teacher_weight = get_teacher_consistency_weight(model_iter_num[idx] // 150)


            ## VNet学生模型与Good_student模型之间的伪标签预测diff_mask_1_2
            diff_mask_1_2 = torch.logical_or(data_arrays[idx].student_predict == 1, data_arrays[Good_student].student_predict == 1).to(torch.int32) - \
                        torch.logical_and(data_arrays[idx].student_predict == 1, data_arrays[Good_student].student_predict == 1).to(torch.int32)

            ## ResVNet学生模型与Good_student模型之间的伪标签预测diff_mask_2_2
            diff_mask_2_2 = torch.logical_or(data_arrays[idx + 2].student_predict == 1, data_arrays[Good_student].student_predict == 1).to(torch.int32) - \
                        torch.logical_and(data_arrays[idx + 2].student_predict == 1, data_arrays[Good_student].student_predict == 1).to(torch.int32)


             # 计算损失与Teacher损失
            if idx != Good_student:
                data_arrays[idx].get_loss(Plabel, consistency_weight, diff_mask_1_2)


            if (idx + 2) != Good_student:
                data_arrays[idx + 2].get_loss(Plabel, consistency_weight, diff_mask_2_2)


            data_arrays[idx].add_teacher_loss(
                teacher_model_array[idx](sampled_batch[0]["image"].cuda()),
                teacher_weight,
            )


            data_arrays[idx + 2].add_teacher_loss(
                teacher_model_array[idx](sampled_batch[0]["image"].cuda()),
                teacher_weight,
            )


            # 记录 teacher_loss 到 TensorBoard
            writer.add_scalar(f"Loss/Teacher_Model_{idx}", data_arrays[idx].teacher_loss.item(), model_iter_num[idx])
            
            # 模型迭代
            
            optimizer_array[idx].zero_grad()
            data_arrays[idx].loss.backward()
            optimizer_array[idx].step()


            optimizer_array[idx + 2].zero_grad()
            data_arrays[idx + 2].loss.backward()
            optimizer_array[idx + 2].step()

            
            # Teacher 迭代
            
            update_ema_variables(
                model_array[idx], teacher_model_array[idx], ema_decay, model_iter_num[idx]
            )


            # 遍历 data_arrays 数组，记录每个模型的损失
            for i, data in enumerate(data_arrays):
                
                # 调用 get_loss(Plabel) 计算并设置损失
                data.get_loss(Plabel, diff_mask)
                

                writer.add_scalar(f"Loss/Consistency_Loss_Model_{i}", data.consistency_loss.item(), model_iter_num[idx])


            # 记录迭代次数
            
            model_iter_num[idx] += 1
    
    for i, model in enumerate(model_array):
        if model_iter_num[idx] >= 3000 and model_iter_num[idx] % len(trainloader) == 0:
            os.makedirs(snapshot_path + "outputs" + f"/{timestamp}", exist_ok=True)
            
            save_mode_path_vnet = os.path.join(
                snapshot_path, "outputs", f"{timestamp}","pmt_" + str(i) + "_iter_" + str(model_iter_num[idx]) + ".pth"
            )
            # 
            
            torch.save(model.state_dict(), save_mode_path_vnet)
            logging.info("save model to {}".format(save_mode_path_vnet))

    # pdb.set_trace()        
    # 比较模型性能，只取最好性能
    global best_score
    for model in model_array:
        model.eval()
    metric = test_calculate_metric(model_array)
    print(metric)

    # 将 metric 的各个元素记录到 TensorBoard
    writer.add_scalar("Metric/Dice", metric[0], model_iter_num[idx])
    writer.add_scalar("Metric/JC", metric[1], model_iter_num[idx])
    writer.add_scalar("Metric/HD", metric[2], model_iter_num[idx])
    writer.add_scalar("Metric/ASD", metric[3], model_iter_num[idx])
    
    

    if metric[0] > best_score:
        
        best_score = metric[0]
        for i, model in enumerate(model_array):

            
            os.makedirs(snapshot_path + "/outputs" + f"/{timestamp}", exist_ok=True)
            
            save_mode_path_vnet = os.path.join(
                snapshot_path, "outputs", f"{timestamp}","pmt_" + str(i) + "_iter_" + str(model_iter_num[idx]) + "_best" +  ".pth"
            )
            
            
            torch.save(model.state_dict(), save_mode_path_vnet)
            logging.info("save model to {}".format(save_mode_path_vnet))


    for model in model_array:
        model.train()


# 主函数

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if not os.path.exists(snapshot_path + "/logs"):
        os.makedirs(snapshot_path + "/logs")
    if os.path.exists(snapshot_path + "/code"):
        shutil.rmtree(snapshot_path + "/code")
    shutil.copytree(
        ".", snapshot_path + "/code", shutil.ignore_patterns([".git", "__pycache__"])
    )

    logging.basicConfig(
        filename=snapshot_path + "/logs" + f"/log_{timestamp}.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(name="vnet"):
        # Network definition
        if name == "vnet":
            net = VNet(
                n_channels=1,
                n_classes=num_classes,
                normalization="batchnorm",
                has_dropout=True,
            )


        elif name == "resVnet":
            net = ResVNet(
                n_channels=1,
                n_classes=num_classes,
                normalization="batchnorm",
                has_dropout=True,
            )


        model = net.cuda()
        return model

    model_array = []
    teacher_model_array = []
    for i in range(model_num):
        model_array.append(create_model(name="vnet"))
        teacher_model_array.append(create_model(name="vnet"))
        for param in teacher_model_array[-1].parameters():
            param.detach_()


    model_array.append(create_model(name="resVnet"))
    model_array.append(create_model(name="resVnet"))


    db_train = LAHeart(
        base_dir=train_data_path,
        split="train",
        train_flod="train.list",  # todo change training flod
        sp_transform=transforms.Compose(
            [
                RandomRotFlip(),
                ToTensor(),
            ]
        ),
    )

    
    labeled_idxs = list(range(args.labelnum * 15))  # todo set labeled num
    unlabeled_idxs = list(range(args.labelnum * 15, (args.labelnum + args.unlabelnum) * 15 ))  # todo set labeled num all_sample_num


    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs
    )
    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        
        
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    optimizer_array = []
    for i in range(model_num):
        optimizer_array.append(
            optim.SGD(
                model_array[i].parameters(),
                lr=base_lr,
                momentum=0.9,
                weight_decay=args.wd,
            )
        )


    optimizer_array.append(
        optim.SGD(
            model_array[2].parameters(),
            lr=base_lr,
            momentum=0.9,
            weight_decay=args.wd,
        )
    )


    optimizer_array.append(
        optim.SGD(
            model_array[3].parameters(),
            lr=base_lr,
            momentum=0.9,
            weight_decay=args.wd,
        )
    )    


    if args.consistency_type == "mse":
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == "kl":
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    iter_num = 0
    
    
    print("The length of trainloader " + str(len(trainloader)))  # 82
    
    epoch_step = len(trainloader) * 2
    model_step = epoch_step // model_num
    
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    for i in range(model_num):
        model_array[i].train()
        teacher_model_array[i].train()


    model_array[2].train()
    model_array[3].train()


    model_iterations = []
    for i in range(model_num):
        model_iterations.append(0)
    data_buffer = []
    training_wl = []
    model_iter_num = [0] * model_num
    for i in range(model_num):
        training_wl.append(i)

    for epoch_num in tqdm(range(max_epoch * len(trainloader) // model_step), ncols=70):
        for i in range(model_step // len(trainloader)):
            for i_batch, sampled_batch in enumerate(trainloader):
                time2 = time.time()
                data_buffer.append(sampled_batch)
                
                ## change lr
                lr_ = get_lr(iter_num, args.warmup_iters, args.thred, max_iterations, base_lr, args.min_lr)
                
                for optimizer in optimizer_array:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr_

                if iter_num >= max_iterations:
                    break
                iter_num = iter_num + 1
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            training_wl.pop(0)
            break
        train_model(
            model_array,
            teacher_model_array,
            optimizer_array,
            data_buffer,
            model_iter_num,
            training_wl[0],
            model_is_first_term[training_wl[0]],
        )
        model_is_first_term[training_wl[0]] = False
        temp_idx = training_wl.pop(0)
        training_wl.append(temp_idx)
        if len(data_buffer) >= epoch_step:
            for i in range(model_step):
                data_buffer.pop(0)
    for i in training_wl:
        train_model(
            model_array,
            teacher_model_array,
            optimizer_array,
            data_buffer,
            model_iter_num,
            i,
            model_is_first_term[i],
        )
    for i, model in enumerate(model_array):
        
        output_folder = os.path.join(snapshot_path, "outputs", f"{timestamp}" , "Final")

        # 创建文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        save_mode_path_vnet = os.path.join(
            snapshot_path, f"outputs", f"{timestamp}", "Final", "pmt_" + str(i) + "_iter_" + str(max_iterations) + ".pth"
        )
        torch.save(model.state_dict(), save_mode_path_vnet)
        logging.info("save model to {}".format(save_mode_path_vnet))

    writer.close()