import h5py
import math
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
import nibabel as nib

import SimpleITK as sitk
from skimage.measure import label

from skimage import morphology
from scipy import ndimage


def label_rescale(image_label, w_ori, h_ori, z_ori, flag):
    w_ori, h_ori, z_ori = int(w_ori), int(h_ori), int(z_ori)
    # resize label map (int)
    if flag == 'trilinear':
        teeth_ids = np.unique(image_label)
        image_label_ori = np.zeros((w_ori, h_ori, z_ori))
        image_label = torch.from_numpy(image_label).cuda(0)
        for label_id in range(len(teeth_ids)):
            image_label_bn = (image_label == teeth_ids[label_id]).float()
            image_label_bn = image_label_bn[None, None, :, :, :]
            image_label_bn = torch.nn.functional.interpolate(image_label_bn, size=(w_ori, h_ori, z_ori),
                                                             mode='trilinear', align_corners=False)
            image_label_bn = image_label_bn[0, 0, :, :, :]
            image_label_bn = image_label_bn.cpu().data.numpy()
            image_label_ori[image_label_bn > 0.5] = teeth_ids[label_id]
        image_label = image_label_ori

    if flag == 'nearest':
        image_label = torch.from_numpy(image_label).cuda(0)
        image_label = image_label[None, None, :, :, :].float()
        image_label = torch.nn.functional.interpolate(image_label, size=(w_ori, h_ori, z_ori), mode='nearest')
        image_label = image_label[0, 0, :, :, :].cpu().data.numpy()
    return image_label



def img_crop(image_bbox):
    if image_bbox.sum() > 0:

        
        x_min = np.nonzero(image_bbox)[0].min() - 8
        x_max = np.nonzero(image_bbox)[0].max() + 8

        y_min = np.nonzero(image_bbox)[1].min() - 16
        y_max = np.nonzero(image_bbox)[1].max() + 16

        z_min = np.nonzero(image_bbox)[2].min() - 16
        z_max = np.nonzero(image_bbox)[2].max() + 16
    

        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if z_min < 0:
            z_min = 0
        if x_max > image_bbox.shape[0]:
            x_max = image_bbox.shape[0]
        if y_max > image_bbox.shape[1]:
            y_max = image_bbox.shape[1]
        if z_max > image_bbox.shape[2]:
            z_max = image_bbox.shape[2]

        if (x_max - x_min) % 16 != 0:  # 防止跳跃连接时尺寸不等
            x_max -= (x_max - x_min) % 16
        if (y_max - y_min) % 16 != 0:
            y_max -= (y_max - y_min) % 16
        if (z_max - z_min) % 16 != 0:
            z_max -= (z_max - z_min) % 16


    if image_bbox.sum() == 0:
        x_min, x_max, y_min, y_max, z_min, z_max = -1, image_bbox.shape[0], 0, image_bbox.shape[1], 0, image_bbox.shape[
            2]
    return x_min, x_max, y_min, y_max, z_min, z_max


def roi_extraction(image, net_roi, ids):
    w, h, d = image.shape
    # roi binary segmentation parameters, the input spacing is 0.4 mm
    print('---run the roi binary segmentation.')
    

    stride_xy = 32
    stride_z = 16
    patch_size_roi_stage = (112, 112, 80)
    
    
    label_roi = roi_detection(net_roi, image[0:w:2, 0:h:2, 0:d:2], stride_xy, stride_z,
                              patch_size_roi_stage)  # (400,400,200)
    print(label_roi.shape, np.max(label_roi))
    label_roi = label_rescale(label_roi, w, h, d, 'trilinear')  # (800,800,400)
    

    label_roi = morphology.remove_small_objects(label_roi.astype(bool), 5000, connectivity=3).astype(float)
    
    label_roi = ndimage.grey_dilation(label_roi, size=(5, 5, 5))
    
    
    label_roi = morphology.remove_small_objects(label_roi.astype(bool), 400000, connectivity=3).astype(float)  # 去除小于2000的连通域
    
    
    label_roi = ndimage.grey_erosion(label_roi, size=(5, 5, 5))
    

    # crop image
    x_min, x_max, y_min, y_max, z_min, z_max = img_crop(label_roi)
    if x_min == -1:  # non-foreground label
        whole_label = np.zeros((w, h, d))
        return whole_label
    image = image[x_min:x_max, y_min:y_max, z_min:z_max]  # (w2,d2,h2),牙齿边界框尺寸
    print("image shape(after roi): ", image.shape)

    return image, x_min, x_max, y_min, y_max, z_min, z_max


def roi_detection(net, image, stride_xy, stride_z, patch_size):
    w, h, d = image.shape  # (400,400,200)

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1  # 2
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1  # 2
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1  # 2
    score_map = np.zeros((2,) + image.shape).astype(np.float32)  # 全0数组(2,400,400,200)
    cnt = np.zeros(image.shape).astype(np.float32)  # 全0数组(400,400,200)
    count = 0
    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1],
                             zs:zs + patch_size[2]]  # 取patch（256,256,160）
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(
                    np.float32)  # 增加两维（1,1,256,256,160）
                test_patch = torch.from_numpy(test_patch).cuda(0)
                with torch.no_grad():
                    y1 = net(test_patch)  # （1,2,256,256,160）
                y = F.softmax(y1, dim=1)  # （1,2,256,256,160）
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]  # （2,256,256,160）
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1],
                      zs:zs + patch_size[2]] + y  # (2,400,400,200)
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1  # (400,400,200)
                count = count + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)


    label_map = np.argmax(score_map, axis=0)  # (400,400,200),0/1
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map


label_index_vec={
    0:np.array([[1,0],[0,0]],dtype=np.float32),
    1:np.array([[0,1],[0,0]],dtype=np.float32),
    2:np.array([[0,0],[1,0]],dtype=np.float32),
    3:np.array([[0,0],[0,1]],dtype=np.float32)
}

def test_all_case(vnet,resnet, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=64, stride_z=32, save_result=True, test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    for image_path in tqdm(image_list):
        id = image_path.split('/')[-1]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        
        if preproc_fn is not None:
            image = preproc_fn(image)


        prediction, score_map = test_single_case(vnet,resnet, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")
    avg_metric = total_metric / len(image_list)
    # print('average metric is {}'.format(avg_metric))

    return avg_metric

def test_all_case_array(model_array, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=64, stride_z=32, save_result=True, test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    for image_path in tqdm(image_list):
        # id = image_path.split('/')[-1]
        id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case_array(model_array, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")
    avg_metric = total_metric / len(image_list)
    # print('average metric is {}'.format(avg_metric))

    return avg_metric

def test_all_case_array_roi(net_roi, model_array, image_list, num_classes, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, save_result=True, test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    for image_path in tqdm(image_list):
        # id = image_path.split('/')[-1]
        id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        
        w, h, d = image.shape

        image, x_min, x_max, y_min, y_max, z_min, z_max = roi_extraction(image, net_roi, id)


        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case_array(model_array, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        prediction = morphology.remove_small_objects(prediction.astype(bool), 3000, connectivity=3).astype(float)
        
        new_prediction = np.zeros((w, h, d))
        new_prediction[x_min:x_max, y_min:y_max, z_min:z_max] = prediction


        if np.sum(new_prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(new_prediction, label[:])
        total_metric += np.asarray(single_metric)
        
        nib.save(nib.Nifti1Image(new_prediction.astype(np.float32),
                                    np.eye(4)), test_save_path + "/{}_pred.nii.gz".format(id))
        nib.save(nib.Nifti1Image(image[:].astype(np.float32),
                                    np.eye(4)), test_save_path + "/{}_img.nii.gz".format(id))
        nib.save(nib.Nifti1Image(label[:].astype(np.float32),
                                    np.eye(4)), test_save_path + "/{}_lab.nii.gz".format(id))

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    return avg_metric

def save_all_case_array(model_array, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=64, stride_z=32, save_result=True, test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    for image_path in tqdm(image_list):
        id = image_path.split('/')[-1]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case_array(model_array, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")
    avg_metric = total_metric / len(image_list)
    # print('average metric is {}'.format(avg_metric))

    return avg_metric

def test_single_case(vnet,resnet=None, image=None, stride_xy=None, stride_z=None, patch_size=None, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)

    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                if vnet is not None:
                    y1 = vnet(test_patch)
                    y  = F.softmax(y1, dim=1)
                    y = y.cpu().data.numpy()
                    y = y[0,:,:,:,:]
                if resnet is not None:
                    y2 = resnet(test_patch)
                    y2 = F.softmax(y2, dim=1)
                    y2 = y2.cpu().data.numpy()
                    if vnet is not None:
                        y = (y+y2[0, :, :, :, :])/2
                    else:
                        y = y2[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    
    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def test_single_case_array(model_array, image=None, stride_xy=None, stride_z=None, patch_size=None, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)

    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                for model in model_array:
                    output = model(test_patch)
                    y_temp = F.softmax(output, dim=1)
                    y_temp = y_temp.cpu().data.numpy()
                    y += y_temp[0,:,:,:,:]
                y /= len(model_array)
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)

    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction==i)
        label_tmp = (label==i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd