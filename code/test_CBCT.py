import os
import argparse
import torch
from networks.vnet import VNet
from test_util import test_all_case_array_roi
from networks.vnet_roi import VNet_roi


from networks.ResVNet import ResVNet


parser = argparse.ArgumentParser()


parser.add_argument(
    "--root_path",
    type=str,
    default="/hy-tmp/dataset/CTooth_13_109/CTooth_data",
    help="Name of Experiment",
)  # todo change dataset path


parser.add_argument(
    "--model", type=str, default="RAIL_CTooth_13_109", help="model_name"
)  # todo change test model name
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--model_num", type=int, default=2, help="num of models")
parser.add_argument("--best_iter", type=int, default=6825, help="num of best training iteration")
parser.add_argument("--logs_time", type=str, default="2025-03-10_18-42-37", help="the log file path name")

FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
snapshot_path = "../model/" + FLAGS.model + "/"
snapshot_path = os.path.abspath(snapshot_path)
test_save_path = "../model/prediction/" + FLAGS.model + "_post/"
model_num = FLAGS.model_num
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(FLAGS.root_path + "/" + "../Flods/test1.list", "r") as f:  # todo change test flod
    
    image_list = f.readlines()
image_list = [

FLAGS.root_path + '/' + item.replace("\n", "") + "/CBCT_roi.h5" for item in image_list

]


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


def test_calculate_metric(epoch_num):
    
    num_classes = 2
    

    net_roi = VNet_roi(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = os.path.join("/hy-tmp/roi_train/model/CTooth_vnet_FDDI_19extraROI_195/vnet/vnet_best_model.pth")
    net_roi.load_state_dict(torch.load(save_mode_path))
    net_roi = net_roi.cuda()  # 确保 net_roi 在 GPU 上
    print("init weight from {}".format(save_mode_path))
    net_roi.eval()
    
    
    model_array = []
    for i in range(4):
        if i == 0 or i == 1:
            model_array.append(create_model(name="vnet"))
        elif i == 2 or i == 3:
            model_array.append(create_model(name="resVnet"))

        model_save_path = os.path.join(
            snapshot_path, "outputs", FLAGS.logs_time, "pmt_" + str(i) + "_iter_" + str(epoch_num) + ".pth"
        )
        
        model_array[i].load_state_dict(torch.load(model_save_path))
        model_array[i] = model_array[i].cuda()
        model_array[i].eval()


    avg_metric = test_all_case_array_roi(
        net_roi,
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


if __name__ == "__main__":
    iters = FLAGS.best_iter
    metric = test_calculate_metric(iters)
    print("iter:", iters)
    print(metric)