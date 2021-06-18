import os
import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings("ignore")

import torch
from torchvision_wj.models.detection.default_retinanet_backbone import * 
from torchvision_wj.models.detection.retinanet import Retinanet
from torchvision_wj.models.segmentation.default_unet_net_vgg import *
from torchvision_wj.models.segwithbox.unetwithbox import UNetWithBox
from torchvision_wj.models.segwithbox.default_unetwithbox_det_vgg import *
from torchvision_wj.models.segwithbox.unetwithbox_det import UNetWithBox_det
from torchvision_wj.utils.losses import *
from torchvision_wj.utils import config, utils
from torchvision_wj.utils.glaucoma_utils import get_glaucoma
from torchvision_wj.utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
import torchvision_wj.utils.transforms as T
from skimage import measure   

def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


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


def calculate_pd_cdr_det(detection_outputs):
    if detection_outputs["boxes"].shape[0]<2:
        pd_vh = [1, 1]
    else:
        boxes = []
        for c in range(2):
            flag = detection_outputs["labels"] == c
            box = detection_outputs["boxes"][flag, :]
            box = box.mean(dim=0)
            boxes.append(box)
        boxes = torch.stack(boxes, dim=0)
        # detection_outputs["boxes"] = boxes
        if boxes.shape[0]==2:
            h_w = boxes[:,2] - boxes[:,0]
            v_w = boxes[:,3] - boxes[:,1]
            pd_vh = [(v_w[0]/v_w[1]).item(), (h_w[0]/h_w[1]).item()]
        else:
            pd_vh = [1, 1]
    return pd_vh, boxes


def calculate_pd_cdr_seg(segmentation_outputs, targets, threshold, smooth=1e-6):
    threshold = np.array(threshold)
    threshold = threshold[:, None, None]
    gt = targets["masks"].bool().cpu().numpy()
    segmentation_outputs = segmentation_outputs[0,:,:gt.shape[1],:gt.shape[2]]
    segmentation_outputs = segmentation_outputs.cpu().numpy()
    pred = segmentation_outputs > threshold
    if pred[0].sum()>0:
        pred[0] = getLargestCC(pred[0])
    if pred[1].sum()>0:
        pred[1] = getLargestCC(pred[1])
    intersect = pred&gt
    pd_dice = (2*np.sum(intersect,axis=(1,2))+smooth)/ \
        (np.sum(pred,axis=(1,2))+np.sum(gt, axis=(1,2))+smooth)

    # CDR pd
    pd_cup, pd_disc = pred[0], pred[1]
    cup = np.argwhere(pd_cup)
    disc = np.argwhere(pd_disc)
    if pd_cup.sum()==0:
        cup_hw = np.zeros((2,))
    else:
        cup_hw = cup.max(axis=0) - cup.min(axis=0) + 1
    if pd_disc.sum()==0:
        cup_hw = np.zeros((2,))
    else:
        disc_hw = disc.max(axis=0) - disc.min(axis=0) + 1 # height+width
    pd_vh_seg = cup_hw/(disc_hw+smooth) # vertical+horizontal 

    return pd_dice, pd_vh_seg

@torch.no_grad()
def detection_segmentation_evaluate(net_name, model, data_loader, threshold, device, save_file):
    torch.set_num_threads(1)
    model.eval()
    gt_cdr_vh_box, gt_cdr_vh_mask = [], []
    pd_cdr_vh_det, pd_cdr_vh_seg, dice_2d = [], [], []
    error = []
    detection_outputs, segmentation_outputs = None, None
    for images, targets in data_loader: 
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        _, outputs = model(images, targets)

        targets = targets[0]
        gt_vh_box, gt_vh_mask = calculate_gt_cdr(targets)
        gt_cdr_vh_box.append(gt_vh_box)
        gt_cdr_vh_mask.append(gt_vh_mask)

        if (net_name == 'retinanet')|(net_name == 'retinanet_unet'):
            detection_outputs = outputs[0]
        elif 'unet_mil' in net_name:
            segmentation_outputs = outputs[0]
        elif ('retinanet_mil' in net_name) | (net_name == 'unet_det_mil'):
            detection_outputs, segmentation_outputs = outputs
            detection_outputs = detection_outputs[0]
            segmentation_outputs = segmentation_outputs[0]

        if detection_outputs is not None:
            pd_vh_det, pd_boxes = calculate_pd_cdr_det(detection_outputs)
            pd_cdr_vh_det.append(pd_vh_det)
            err = torch.abs(pd_boxes-targets["boxes"]).sum(dim=1)
            error.append(err.cpu().numpy())
            # pd_wh = pd_boxes[:,2:] - pd_boxes[:,:2]
            # gt_wh = targets["boxes"][:,2:] - targets["boxes"][:,:2]
            # err_wh = torch.abs(pd_wh-gt_wh)/gt_wh
        if segmentation_outputs is not None:
            pd_dice, pd_vh_seg = calculate_pd_cdr_seg(segmentation_outputs, targets,
                                                    threshold)
            pd_cdr_vh_seg.append(pd_vh_seg)
            dice_2d.append(pd_dice)
    gt_cdr_vh_box = np.vstack(gt_cdr_vh_box)
    gt_cdr_vh_mask = np.vstack(gt_cdr_vh_mask)
    results = np.hstack([gt_cdr_vh_box, gt_cdr_vh_mask])
    columns = ['gt_vcdr_bbox', 'gt_hcdr_bbox', 'gt_vcdr_mask', 'gt_hcdr_mask']
    if detection_outputs is not None: 
        pd_cdr_vh_det = np.vstack(pd_cdr_vh_det)
        error = np.vstack(error)
        results = np.hstack([results, pd_cdr_vh_det, error])
        columns += ['pd_vcdr_det', 'pd_hcdr_det', 'cup_error', 'disc_error']
    if segmentation_outputs is not None:
        pd_cdr_vh_seg = np.vstack(pd_cdr_vh_seg)
        dice_2d = np.vstack(dice_2d)
        results = np.hstack([results, pd_cdr_vh_seg, dice_2d])
        columns += ['pd_vcdr_seg', 'pd_hcdr_seg', 'cup_dice', 'disc_dice']
    results = pd.DataFrame(data=results, columns=columns)
    results.to_csv(save_file, index=False)

    if segmentation_outputs is not None: 
        print("segmentation output")
        mean_2d = dice_2d.mean(axis=0)
        std_2d = dice_2d.std(axis=0)
        print(f'segmentation dice: {mean_2d}({std_2d})')
        diff = np.abs(pd_cdr_vh_seg - gt_cdr_vh_mask)
        diff_mean = diff.mean(axis=0)
        diff_std = diff.std(axis=0)
        print(f'segmentation cdr error (mask): {diff_mean}({diff_std})')
    if detection_outputs is not None:
        print("detection output")
        diff = np.abs(pd_cdr_vh_det - gt_cdr_vh_mask)
        diff_mean = diff.mean(axis=0)
        diff_std = diff.std(axis=0)
        print(f'detection cdr error (mask): {diff_mean}({diff_std})')
        diff = np.abs(pd_cdr_vh_det - gt_cdr_vh_box)
        diff_mean = diff.mean(axis=0)
        diff_std = diff.std(axis=0)
        print(f'detection cdr error (box): {diff_mean}({diff_std})')
        mean = error.mean(axis=0)
        std = diff.std(axis=0)
        print(f'detection width/height error (box): {mean}({std})')


@torch.no_grad()
def segmentation_valid_evaluate(net_name, model, data_loader, device, threshold, 
        save_detection=None, smooth=1e-10):
    file_cup_dice = os.path.join(save_detection,'cup_dice_2d.xlsx')    
    file_disc_dice = os.path.join(save_detection,'disc_dice_2d.xlsx')    
    torch.set_num_threads(1)
    model.eval()
    dice_2d  = {k: [] for k in threshold}
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        gt = torch.stack([t["masks"] for t in targets], dim=0)
        gt = gt.bool().cpu().numpy()
        _, outputs = model(images, targets)
        if 'unet_mil' in net_name:
            out = outputs[0]
        elif ('retinanet_mil' in net_name) | (net_name == 'unet_det_mil'):
            out = outputs[1][0]
        out = out[:,:,:gt.shape[2],:gt.shape[3]]
        out = out.cpu().numpy()
        for n_th,th in enumerate(threshold):
            pred = out>th
            if pred[0,0].sum()>0:
                pred[0,0] = getLargestCC(pred[0,0])
            if pred[0,1].sum()>0:
                pred[0,1] = getLargestCC(pred[0,1])
            intersect = pred&gt
            v_dice_2d = (2*np.sum(intersect,axis=(0,2,3))+smooth)/ \
                (np.sum(pred,axis=(0,2,3))+np.sum(gt,axis=(0,2,3))+smooth)
            dice_2d[th].append(v_dice_2d)
    dice_2d = [np.vstack(dice_2d[key]) for key in dice_2d.keys()]
    dice_2d = np.stack(dice_2d, axis=0)#np.vstack(dice_2d).T
    cup_2d =  pd.DataFrame(data=dice_2d[:,:,0].T, columns=threshold)
    disc_2d = pd.DataFrame(data=dice_2d[:,:,1].T, columns=threshold)
    cup_2d.to_excel(file_cup_dice, index=False)
    disc_2d.to_excel(file_disc_dice, index=False)

    mean_2d = np.mean(dice_2d, axis=1)
    std_2d = np.std(dice_2d, axis=1)
    loc2 = np.argmax(mean_2d, axis=0)
    print('2d mean: {}({})'.format(mean_2d[loc2,torch.arange(2)],std_2d[loc2,torch.arange(2)]))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_exp', default=12, type=int,
                        help='the index of experiments')
    # parser.add_argument('--eval_mode', default='valid', type=str,
    #                     help='the index of experiments')
    # parser.add_argument('--n_dataset', default=0, type=int,
    #                     help='the index of experiments')
    # distributed training parameters
    parser.add_argument('--world-size', default=12, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print(args)

    n_exp = args.n_exp
    dir_save_root = os.path.join('results', 'glaucoma_retinanet')
    if n_exp == 0:
        net_name = 'retinanet'
        experiment_names = ['retinanet_B']
    elif n_exp == 1:
        net_name = 'unet_mil'
        experiment_names = ['unet_BB_supervised']
    elif n_exp == 2:
        net_name = 'unet_mil'
        experiment_names = ['unet_mil_BB', 'unet_mil_BB_10', 'unet_mil_BB_20', 'unet_mil_BB_30']
    elif n_exp == 3:
        net_name = 'unet_mil'
        experiment_names = ['unet_mil_BB_10_expsumr=6', 'unet_mil_BB_10_expsumr=8',
                            'unet_mil_BB_10_expsumr=10', 'unet_mil_BB_10_explogs=6', 
                            'unet_mil_BB_10_explogs=8', 'unet_mil_BB_10_explogs=10']
    elif n_exp == 4:
        net_name = 'unet_det_mil'
        experiment_names = ['unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=3',
                            'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=4',
                            'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=5',
                            'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=6',
                            'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=7']
    elif n_exp == 5:
        net_name = 'unet_det_mil'
        experiment_names = ['unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=6_iou_T=0.5',
                            'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=6_iou_T=0.6',
                            'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=6_iou_T=0.7',
                            'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=6_iou_T=0.5',
                            'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=6_iou_T=0.6',
                            'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=6_iou_T=0.7']
    elif n_exp == 6:
        net_name = 'unet_det_mil'
        experiment_names = ['unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=3_iou_T=0.6',
                            'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=4_iou_T=0.6',
                            'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=5_iou_T=0.6',
                            'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=7_iou_T=0.6',
                            'unet_det_mil_BB_10_dconv=6_expsumr=8_L1sigma=8_iou_T=0.6']
    elif n_exp == 7:
        net_name = 'unet_det_mil'
        experiment_names = ['unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=3_iou_T=0.5',
                            'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=4_iou_T=0.5',
                            'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=5_iou_T=0.5',
                            'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=7_iou_T=0.5',
                            'unet_det_mil_BB_10_dconv=6_explogs=8_L1sigma=8_iou_T=0.5']
        
    
    for experiment_name in experiment_names:
        output_dir = os.path.join(dir_save_root, experiment_name)
        print(output_dir)
        _C = config.read_config_file(os.path.join(output_dir, 'config.yaml'))
        assert _C['save_params']['experiment_name']==experiment_name, "experiment_name is not right"
        cfg = {'detector_params': {'detections_per_class': 1}, 'workers': 4}
        _C = config.config_updates(_C, cfg)

        train_valid_summary = pd.read_csv(os.path.join(output_dir, "summary.csv"))
        train_valid_summary['val_loss'][:20] = train_valid_summary['val_loss'][:20] + 100
        loc = train_valid_summary['val_loss'].argmin()
        epoch = train_valid_summary['epoch'].loc[loc]
        valid_loss = train_valid_summary['val_loss'].min()
        assert epoch == train_valid_summary['epoch'].iloc[-9]
        print(f"epoch = {epoch}, valid_loss = {valid_loss}")

        train_params       = _C['train_params']
        data_params        = _C['data_params']
        net_params         = _C['net_params']
        detector_params    = _C['detector_params']
        dataset_params     = _C['dataset']
        save_params        = _C['save_params']
        data_visual_aug    = data_params['data_visual_aug']
        data_transform_aug = data_params['data_transform_aug']

        device = torch.device(_C['device'])

        ## set up models
        print("Creating model with parameters: {}".format(net_params))
        losses, loss_weights = dict(), dict()
        for task_name, task_losses_list in net_params['losses'].items():
            task_losses, task_loss_weights = [], []
            for loss in task_losses_list:
                task_losses.append(eval(loss[0])(**loss[1]))
                task_loss_weights.append(loss[2])
            losses[task_name] = task_losses
            loss_weights[task_name] = task_loss_weights
        backbone_params = net_params['backbone_params']
        backbone_arch = backbone_params.pop('model_name')
        backbone = eval(backbone_arch)(**backbone_params)
        if (net_name == 'retinanet') | (net_name == 'retinanet_unet'):
            model = Retinanet(backbone, losses['detection_losses'], 
                            loss_weights['detection_losses'], **detector_params) 
        elif net_name == 'unet_mil':
            model = UNetWithBox(backbone, losses['segmentation_losses'], 
                            loss_weights['segmentation_losses'], softmax=backbone_params['softmax'])
        elif net_name == 'retinanet_mil':
            print("Creating model with detector parameters: {}".format(detector_params))
            model = Retinanet_MIL(backbone, losses, loss_weights, **detector_params) 
        elif net_name == 'retinanet_mil_atts':
            print("Creating model with detector parameters: {}".format(detector_params))
            model = Retinanet_MIL_atts(backbone, losses, loss_weights, **detector_params) 
        elif net_name == 'unet_det_mil':
            model = UNetWithBox_det(backbone, losses, loss_weights, 
                            softmax=backbone_params['softmax'],
                            obj_sizes=net_params['obj_sizes'],
                            detection_sample_selection=net_params['detection_sample_selection'],
                            box_normalizer=net_params['box_normalizer'],
                            nms_score_threshold=detector_params['nms_score_threshold'],
                            nms_iou_threshold=detector_params['nms_iou_threshold'],
                            detections_per_class=detector_params['detections_per_class'])
        model.to(device)

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                find_unused_parameters=True)
            model_without_ddp = model.module

        model_file = 'model_{:02d}'.format(epoch)
        print('loading model {}.pth'.format(model_file))
        checkpoint = torch.load(os.path.join(output_dir, model_file+'.pth'), map_location='cpu')
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint['model'].items():
        #     name = k.replace('module.','') # remove 'module.' of dataparallel
        #     new_state_dict[name]=v
        model_without_ddp.load_state_dict(checkpoint['model'])

        print('start evaluating {} ...'.format(epoch))
        model.training = False

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

        # Data loading code
        print("Loading data")
        threshold_cand = [0.001, 0.005, 0.01] + list(np.arange(0.05, 0.9, 0.05))
        for n_dataset in range(6):
            print(f"dataset = {n_dataset}")
            if n_dataset == 0:
                dataset_name = 'valid'
                dataset_test = get_glaucoma(root=dataset_params['root_path'], csv_file=dataset_params['valid_file'], 
                                            transforms=get_transform(train=False))
            elif n_dataset == 1:
                dataset_name = 'test'
                dataset_test = get_glaucoma(root=dataset_params['root_path'], csv_file=dataset_params['test_file'], 
                                            transforms=get_transform(train=False))
            elif n_dataset == 2:
                dataset_name = 'grader1'
                dataset_test = get_glaucoma(root=dataset_params['root_path'], 
                                            csv_file='dataset_list_grader1_20201017', 
                                            transforms=get_transform(train=False))
            elif n_dataset == 3:
                dataset_name = 'grader2'
                dataset_test = get_glaucoma(root=dataset_params['root_path'], 
                                            csv_file='dataset_list_grader2_20201017', 
                                            transforms=get_transform(train=False))
            elif n_dataset == 4:
                dataset_name = 'grader3'
                dataset_test = get_glaucoma(root=dataset_params['root_path'], 
                                            csv_file='dataset_list_grader3_20201017', 
                                            transforms=get_transform(train=False))
            elif n_dataset == 5:
                dataset_name = 'grader4'
                dataset_test = get_glaucoma(root=dataset_params['root_path'], 
                                            csv_file='dataset_list_grader4_20201017', 
                                            transforms=get_transform(train=False))
            image_names = dataset_test.csv['image_name']
            
            print("Creating data loaders")
            if args.distributed:
                test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
            else:
                test_sampler = torch.utils.data.SequentialSampler(dataset_test)

            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=1,
                sampler=test_sampler, num_workers=_C['workers'],
                collate_fn=utils.collate_fn)

            threshold = None
            save_file = os.path.join(output_dir, f"{dataset_name}_cdr.csv")
            segmentation_threshold = None
            if (net_name == 'unet_mil') | ('retinanet_mil' in net_name) | \
                    (net_name == 'unet_det_mil'):
                if dataset_name == 'valid':
                    threshold = [0.001, 0.005, 0.01] + list(np.arange(0.05, 0.9, 0.05))
                    segmentation_valid_evaluate(net_name, model, data_loader_test, 
                                        device, threshold, save_detection=output_dir)
                cup_2d = pd.read_excel(os.path.join(output_dir, "cup_dice_2d.xlsx")).mean(axis=0)
                threshold_cup = cup_2d.argmax(cup_2d)
                disc_2d = pd.read_excel(os.path.join(output_dir, "disc_dice_2d.xlsx")).mean(axis=0)
                threshold_disc = disc_2d.argmax()
                segmentation_threshold = [threshold_cup, threshold_disc]
            
            detection_segmentation_evaluate(net_name, model, data_loader_test, 
                            segmentation_threshold, device=device, save_file=save_file)
