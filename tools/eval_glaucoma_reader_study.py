import os, sys
import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings("ignore")

import torch
from torchvision_wj.utils import config, utils
from torchvision_wj.utils.glaucoma_utils import get_glaucoma
import torchvision_wj.utils.transforms as T
from skimage.measure import label   


def calculate_gt_cdr(targets):
    # use mask to calculate cdr for fair comparison with the segmentation results
    h_w = targets["boxes"][:,2] - targets["boxes"][:,0]
    v_w = targets["boxes"][:,3] - targets["boxes"][:,1]
    gt_vh_box = [(v_w[0]/v_w[1]).item(), (h_w[0]/h_w[1]).item()]

    gt = targets["masks"]
    gt = gt.bool().cpu().numpy()
    gt_cup, gt_disc = gt[0], gt[1]
    cup = np.argwhere(gt_cup)
    disc = np.argwhere(gt_disc)
    cup_hw = cup.max(axis=0) - cup.min(axis=0) + 1
    disc_hw = disc.max(axis=0) - disc.min(axis=0) + 1 # height+width
    gt_vh_mask = cup_hw/disc_hw

    return gt_vh_box, gt_vh_mask


if __name__ == "__main__":
    dir_save_root = os.path.join('results', 'glaucoma_retinanet')
    experiment_name = 'unet_BB_supervised'
    output_dir = os.path.join(dir_save_root, experiment_name)
    print(output_dir)
    _C = config.read_config_file(os.path.join(output_dir, 'config.yaml'))

    data_params        = _C['data_params']
    dataset_params     = _C['dataset']
    data_visual_aug    = data_params['data_visual_aug']
    data_transform_aug = data_params['data_transform_aug']
    device = torch.device('cpu')
    def get_transform(train):
        transforms = []
        transforms.append(T.ToTensor())
        transforms.append(T.Resize(width=data_params['crop_width']))
        transforms.append(T.Normalizer(mode=data_params['normalizer_mode']))
        if train:
            transforms.append(T.RandomCrop(prob=data_params['crop_ob_prob'], 
                                        crop_size=data_params['crop_size'], 
                                        weak=True))
        return T.Compose(transforms)

    datasets = ['grader1', 'grader2', 'grader3', 'grader4']
    csv_files = ['dataset_list_grader1_20201017', 'dataset_list_grader2_20201017',
                 'dataset_list_grader3_20201017', 'dataset_list_grader4_20201017']
    for k1 in range(len(datasets)):
        dataset1 = datasets[k1]
        csv_file1 = csv_files[k1]
        dataset_test1 = get_glaucoma(root=dataset_params['root_path'], 
                                    csv_file=csv_file1, 
                                    transforms=get_transform(train=False))
        image_names = dataset_test1.csv['image_name']

        test_sampler1 = torch.utils.data.SequentialSampler(dataset_test1)
        data_loader_test1 = torch.utils.data.DataLoader(
                dataset_test1, batch_size=1,
                sampler=test_sampler1, num_workers=0,
                collate_fn=utils.collate_fn)
        for k2 in range(k1+1,len(datasets)):
            dataset2 = datasets[k2]
            csv_file2 = csv_files[k2]
            print(f"<<< {dataset1} vs. {dataset2}")
            dataset_test2 = get_glaucoma(root=dataset_params['root_path'], 
                                        csv_file=csv_file2, 
                                        transforms=get_transform(train=False))
            test_sampler2 = torch.utils.data.SequentialSampler(dataset_test2)
            data_loader_test2 = torch.utils.data.DataLoader(
                    dataset_test2, batch_size=1,
                    sampler=test_sampler2, num_workers=0,
                    collate_fn=utils.collate_fn)
            performance = []
            for data1, data2 in zip(data_loader_test1, data_loader_test2): 
                image1, target1 = data1
                image2, target2 = data2

                mask1 = torch.stack([t["masks"] for t in target1], dim=0)
                mask1 = mask1.bool().cpu().numpy()
                mask1 = mask1[0]

                mask2 = torch.stack([t["masks"] for t in target2], dim=0)
                mask2 = mask2.bool().cpu().numpy()
                mask2 = mask2[0]
                # print(mask1.shape, mask2.shape)

                smooth = 1e-6
                intersect = mask1&mask2
                dice = (2*np.sum(intersect,axis=(1,2))+smooth)/ \
                    (np.sum(mask1,axis=(1,2))+np.sum(mask2, axis=(1,2))+smooth)

                cdr_vh_box1, cdr_vh_mask1 = calculate_gt_cdr(target1[0])
                cdr_vh_box2, cdr_vh_mask2 = calculate_gt_cdr(target2[0])
                perf = np.hstack([cdr_vh_box1, cdr_vh_mask1, cdr_vh_box2, cdr_vh_mask2, dice])
                performance.append(perf)
            performance = np.vstack(performance)
            columns=['vcdr_bbox1','hcdr_bbox1','vcdr_mask1','hcdr_mask1',
                     'vcdr_bbox2','hcdr_bbox2','vcdr_mask2','hcdr_mask2',
                     'cup_dice','disc_dice']
            performance = pd.DataFrame(data=performance, columns=columns)
            performance.to_csv(os.path.join(dir_save_root, f'dataset_{dataset1}_{dataset2}.csv'), index=False)