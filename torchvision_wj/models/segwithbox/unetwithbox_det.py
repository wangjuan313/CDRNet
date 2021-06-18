import copy
import torch
from torch import nn
from torchvision_wj.models.batch_image import BatchImage
from torchvision_wj.utils.losses import *
from torch.jit.annotations import Tuple, List, Dict, Optional
from torchvision.ops import boxes as box_ops

__all__ = ['UNetWithBox_det']

class UNetWithBox_det(nn.Module):
    def __init__(self, model, losses, loss_weights, softmax, obj_sizes, 
                detection_sample_selection={'method': 'all'}, box_normalizer=None,
                nms_score_threshold=0.3, nms_iou_threshold=0.4, detections_per_class=1,
                batch_image=BatchImage, size_divisible=32, post_obj=True):

        super(UNetWithBox_det, self).__init__()        
        self.model = model
        nb_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('##### trainable parameters in model: {}'.format(nb_params))
        self.losses_func = losses
        self.loss_weights = loss_weights
        self.softmax = softmax
        self.batch_image = batch_image(size_divisible)
        self.det_level_in_seg = self.model.det_level_in_seg
        self.obj_sizes = obj_sizes
        self.detection_sample_selection = detection_sample_selection
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.detections_per_class = detections_per_class
        if box_normalizer is not None:
            self.box_normalizer = torch.FloatTensor(box_normalizer)
        else:
            self.box_normalizer = box_normalizer
        self.post_obj = post_obj

    def sigmoid_compute_seg_loss(self, seg_preds, targets, image_shape):
        device = seg_preds[0].device
        dtype  = seg_preds[0].dtype
        all_labels = [t['labels'] for t in targets]
        ytrue = torch.stack([t['masks'] for t in targets],dim=0).long()
        label_unique = torch.unique(torch.cat(all_labels, dim=0))
        seg_losses = {}
        for nb_level in range(len(seg_preds)):
            preds = seg_preds[nb_level]
            stride = image_shape[-1]/preds.shape[-1]

            mask = preds.new_full(preds.shape,0,device=preds.device)
            crop_boxes = []
            gt_boxes   = []
            for n_img, target in enumerate(targets):
                boxes = torch.round(target['boxes']/stride).type(torch.int32)
                labels = target['labels']
                for n in range(len(labels)):
                    box = boxes[n,:]
                    c   = labels[n]#.item()
                    mask[n_img,c,box[1]:box[3]+1,box[0]:box[2]+1] = 1

                    height, width = (box[2]-box[0]+1)/2.0, (box[3]-box[1]+1)/2.0
                    r  = torch.sqrt(height**2+width**2)
                    cx = (box[2]+box[0]+1)//2
                    cy = (box[3]+box[1]+1)//2
                    # print('//// box ////',box, cx, cy, r)
                    crop_boxes.append(torch.tensor([n_img, c, cx, cy, r]))
                    gt_boxes.append(torch.tensor([n_img, c, box[0], box[1], box[2], box[3]], dtype=torch.int32, device=device))
            if len(crop_boxes)==0:
                crop_boxes = torch.empty((0,5), device=device)
            else:
                crop_boxes = torch.stack(crop_boxes, dim=0)
            if len(gt_boxes)==0:
                gt_boxes = torch.empty((0,6), device=device)
            else:
                gt_boxes = torch.stack(gt_boxes, dim=0) 

            # print('#boxes',crop_boxes.shape[0],gt_boxes.shape[0])
            assert crop_boxes.shape[0]==gt_boxes.shape[0]

            kwargs_opt = {'ypred':preds, 'ytrue':ytrue, 'mask':mask, 'gt_boxes':gt_boxes, 'crop_boxes':crop_boxes}
            for loss_func, loss_w in zip(self.losses_func["segmentation_losses"], self.loss_weights["segmentation_losses"]):
                loss_keys = loss_func.__call__.__code__.co_varnames
                loss_params = {key:kwargs_opt[key] for key in kwargs_opt.keys() if key in loss_keys}
                loss_v = loss_func(**loss_params)*loss_w

                key_prefix = type(loss_func).__name__+'/'+str(nb_level)+'/'
                loss_v = {key_prefix+str(n):loss_v[n] for n in range(len(loss_v))}
                seg_losses.update(loss_v)
        
        return seg_losses

    def softmax_compute_seg_loss(self, seg_preds, targets, image_shape, eps=1e-6):
        device = seg_preds[0].device
        dtype  = seg_preds[0].dtype
        all_labels = [t['labels'] for t in targets]
        ytrue = torch.stack([t['masks'] for t in targets],dim=0).long()
        label_unique = torch.unique(torch.cat(all_labels, dim=0))
        seg_losses = {}
        for nb_level in range(len(seg_preds)):
            preds = seg_preds[nb_level]
            stride = image_shape[-1]/preds.shape[-1]

            mask = preds.new_full(preds.shape,0,device=preds.device)
            crop_boxes = []
            gt_boxes   = []
            for n_img, target in enumerate(targets):
                boxes = torch.round(target['boxes']/stride).type(torch.int32)
                labels = target['labels']
                for n in range(len(labels)):
                    box = boxes[n,:]
                    c   = labels[n]#.item()
                    mask[n_img,c,box[1]:box[3]+1,box[0]:box[2]+1] = 1

                    height, width = (box[2]-box[0]+1)/2.0, (box[3]-box[1]+1)/2.0
                    r  = torch.sqrt(height**2+width**2)
                    cx = (box[2]+box[0]+1)//2
                    cy = (box[3]+box[1]+1)//2
                    # print('//// box ////',box, cx, cy, r)
                    crop_boxes.append(torch.tensor([n_img, c, cx, cy, r]))
                    gt_boxes.append(torch.tensor([n_img, c, box[0], box[1], box[2], box[3]], dtype=torch.int32, device=device))
            if len(crop_boxes)==0:
                crop_boxes = torch.empty((0,5), device=device)
            else:
                crop_boxes = torch.stack(crop_boxes, dim=0)
            if len(gt_boxes)==0:
                gt_boxes = torch.empty((0,6), device=device)
            else:
                gt_boxes = torch.stack(gt_boxes, dim=0) 

            # print('#boxes',crop_boxes.shape[0],gt_boxes.shape[0])
            assert crop_boxes.shape[0]==gt_boxes.shape[0]

            kwargs_opt = {'ypred':preds, 'ytrue':ytrue, 'mask':mask, 'gt_boxes':gt_boxes, 'crop_boxes':crop_boxes}
            for loss_func, loss_w in zip(self.losses_func,self.loss_weights):
                loss_keys = loss_func.__call__.__code__.co_varnames
                loss_params = {key:kwargs_opt[key] for key in kwargs_opt.keys() if key in loss_keys}
                loss_v = loss_func(**loss_params)*loss_w

                key_prefix = type(loss_func).__name__+'/'+str(nb_level)+'/'
                loss_v = {key_prefix+str(n):loss_v[n] for n in range(len(loss_v))}
                seg_losses.update(loss_v)

        return seg_losses

    def calculate_expected_iou(self, gt_boxes, anchors_center):
        assert anchors_center.shape[1] == 2
        assert gt_boxes.shape[1] == 4

        left = anchors_center[:, 0] - gt_boxes[:, 0]
        top = anchors_center[:, 1] - gt_boxes[:, 1]
        right = gt_boxes[:, 2] - anchors_center[:, 0]
        bottom = gt_boxes[:, 3] - anchors_center[:, 1]

        height = gt_boxes[:, 3] - gt_boxes[:, 1]
        width = gt_boxes[:, 2] - gt_boxes[:, 0]

        r1 = left/width.float()
        r2 = top/height.float()

        flag = (left >= 0) & (top >= 0) & (right >= 0) & (bottom >= 0)
        r1 = r1*flag
        r2 = r2*flag
        r1 = torch.min(r1, 1-r1)
        r2 = torch.min(r2, 1-r2)

        iou_expected = torch.max(4*r1*r2, 1/(4*(1-r1)*(1-r2)))
        iou_expected = torch.max(iou_expected, 2*r1/(1+2*r1*(1-2*r2)))
        iou_expected = torch.max(iou_expected, 2*r2/(1+2*r2*(1-2*r1)))
        return iou_expected

    def get_grid_centers(self, h, w, stride, device):
        shift_x = torch.arange(0, w, device=device) + (1-1/stride[0])/2.0#* stride[0]
        shift_y = torch.arange(0, h, device=device) + (1-1/stride[1])/2.0#* stride[1]
        # shift_x = torch.arange(0, pred.shape[-1], device=device) + (1-1/stride[0])/2.0#* stride[0]
        # shift_y = torch.arange(0, pred.shape[-2], device=device) + (1-1/stride[1])/2.0#* stride[1]
        center_x = shift_x.view(1, -1).repeat(len(shift_y), 1)
        center_y = shift_y.view(-1, 1).repeat(1, len(shift_x))
        centers = torch.stack([center_x, center_y, center_x, center_y], dim=0)
        return centers

    def get_expected_ious(self, seg_preds, targets, image_shape):
        device = seg_preds[0].device
        expected_ious = []
        for seg in seg_preds:
            ious = seg.new_full(seg.shape, 0) # BxCxHxW
            stride = torch.tensor(image_shape)//torch.tensor(seg.shape[-2:])
            assert stride[0] == stride[1]
            centers = self.get_grid_centers(seg.shape[-2], seg.shape[-1],
                        stride, device)
            for itx, target in enumerate(targets):
                if target["boxes"].shape[0]==0:
                    continue  
                iou_obj = self.calculate_expected_iou(target["boxes"][:,:,None,None]/stride[0], centers[None,:2])
                for k, c in enumerate(target['labels']):
                    ious[itx, c] = torch.max(ious[itx, c], iou_obj[k])
            expected_ious.append(ious)
        return expected_ious

    def get_bbox_regression_targets(self, det_preds, targets, image_shape):
        device = det_preds[0].device
        # num_class = det_preds[0].shape[1]//4
        det_gts, flag_det_gts = [], []
        # print("<<< #dets", len(det_preds))
        for idx,pred in enumerate(det_preds):
            # print("<<< pred", pred.shape)
            det_gt = pred.new_full(pred.shape, 0) #BxCx4xHxW
            flag_det_gt = pred.new_full((pred.shape[0], pred.shape[1], 1,
                    pred.shape[3], pred.shape[4]), 0, dtype=torch.bool) #BxCx1xHxW
            stride = torch.tensor(image_shape)//torch.tensor(pred.shape[-2:])
            assert stride[0] == stride[1]
            centers = self.get_grid_centers(pred.shape[-2], pred.shape[-1],
                        stride, device)
            for itx, target in enumerate(targets):
                # print(f".......{idx}......{itx}")
                if target["boxes"].shape[0]==0:
                    continue    
                diff = centers[None] - target["boxes"][:,:,None,None]/stride[0]
                flag_diff = (diff[:,0]>0)&(diff[:,1]>0)&(diff[:,2]<0)&(diff[:,3]<0)
                if self.box_normalizer is None:
                    normalizer = 2**(len(det_preds)-idx-1)
                else:
                    normalizer = self.box_normalizer[target["labels"]].to(device)/stride[0]
                    normalizer = normalizer[:, None, None, None]
                diff = diff/normalizer
                wh = (target["boxes"][:,2:]-target["boxes"][:,:2])/stride[0]
                flag_obj = (wh[:,0]>self.obj_sizes[0])&(wh[:,0]<self.obj_sizes[1])& \
                        (wh[:,1]>self.obj_sizes[0])&(wh[:,1]<self.obj_sizes[1]) # determined by receptive field
                for k, c in enumerate(target['labels']):
                    # if not flag_obj[k]:
                    #     continue
                    det_gt[itx, c] = diff[k] ## only one object in each class
                    flag_det_gt[itx, c] = flag_diff[k].unsqueeze(dim=0)
                    ##TODO multiple ojects in each class with potential overlapping
            det_gts.append(det_gt)
            flag_det_gts.append(flag_det_gt)
        return det_gts, flag_det_gts

    def bbox_decoder(self, det_preds, image_shape):
        device = det_preds[0].device
        det_preds_decode = []
        for idx,pred in enumerate(det_preds):
            stride = torch.tensor(image_shape)//torch.tensor(pred.shape[-2:])
            assert stride[0] == stride[1]
            centers = self.get_grid_centers(pred.shape[-2], pred.shape[-1], stride, device)
            if self.box_normalizer is None:
                normalizer = 2**(len(det_preds)-idx-1)
            else:
                normalizer = self.box_normalizer.to(device)/stride[0].float()
                normalizer = normalizer[None, :, None, None, None]
            pred_decode = (centers[None] - pred * normalizer) * stride[0]
            det_preds_decode.append(pred_decode)
        return det_preds_decode

    # def get_flatten_boxes(self, boxes_list, flag=False):
    #     if flag:
    #         boxes_list = [boxes.permute(0,1,3,4,2).reshape(-1) for boxes in boxes_list]
    #     else:
    #         boxes_list = [boxes.permute(0,1,3,4,2).reshape(-1,4) for boxes in boxes_list]
    #     boxes_list = torch.cat(boxes_list)
    #     return boxes_list

    def get_flatten_boxes(self, boxes_list, num_classes, flag=False):
        if flag:
            boxes_list = [boxes.permute(1,0,3,4,2).reshape(num_classes, -1) for boxes in boxes_list]
        else:
            boxes_list = [boxes.permute(1,0,3,4,2).reshape(num_classes, -1, 4) for boxes in boxes_list]
        boxes_list = torch.cat(boxes_list, dim=1)
        return boxes_list

    def get_topk_predictions(self, topk, seg_preds, flag_det_gts, det_gts, det_preds, det_gts_decode, det_preds_decode):
        seg_preds = [seg_preds[k] for k in self.det_level_in_seg]
        bs, num_classes = seg_preds[0].shape[:2]
        det_gts_dict, det_preds_dict = {k:[]for k in range(num_classes)}, {k:[]for k in range(num_classes)}
        det_gts_decode_dict, det_preds_decode_dict = {k:[]for k in range(num_classes)}, {k:[]for k in range(num_classes)}
        for k in range(len(seg_preds)):
            flag_det_gt = flag_det_gts[k].permute(0,1,3,4,2).reshape(bs, num_classes, -1) #BxCx(MN)
            seg_pred = seg_preds[k].unsqueeze(2).permute(0,1,3,4,2).reshape(bs, num_classes, -1) #BxCx(MN)
            det_gt = det_gts[k].permute(0,1,3,4,2).reshape(bs, num_classes, -1, 4) #BxCx(MN)x4
            det_pred = det_preds[k].permute(0,1,3,4,2).reshape(bs, num_classes, -1, 4) #BxCx(MN)x4
            det_gt_decode = det_gts_decode[k].permute(0,1,3,4,2).reshape(bs, num_classes, -1, 4) #BxCx(MN)x4
            det_pred_decode = det_preds_decode[k].permute(0,1,3,4,2).reshape(bs, num_classes, -1, 4) #BxCx(MN)x4
            for n in range(bs):
                nb =  flag_det_gt[n].sum(dim=1)
                if (nb[0]==0) | (nb[1]==0):
                    continue
                for c in range(num_classes):
                    flag = flag_det_gt[n,c]
                    values, topk_idxs = (seg_pred[n,c,flag]).topk(topk, dim=0, largest=False)
                    det_gts_dict[c].append(det_gt[n,c,flag][topk_idxs])
                    det_preds_dict[c].append(det_pred[n,c,flag][topk_idxs])
                    det_gts_decode_dict[c].append(det_gt_decode[n,c,flag][topk_idxs])
                    det_preds_decode_dict[c].append(det_pred_decode[n,c,flag][topk_idxs])
        det_gts_dict = {key: torch.cat(values, dim=0) for key, values in det_gts_dict.items()} 
        det_preds_dict = {key: torch.cat(values, dim=0) for key, values in det_preds_dict.items()} 
        det_gts_decode_dict = {key: torch.cat(values, dim=0) for key, values in det_gts_decode_dict.items()} 
        det_preds_decode_dict = {key: torch.cat(values, dim=0) for key, values in det_preds_decode_dict.items()} 
        return det_gts_dict, det_preds_dict, det_gts_decode_dict, det_preds_decode_dict
    
    def get_all_predictions(self, flag_det_gts, det_gts, det_preds, det_gts_decode, det_preds_decode):
        num_classes = flag_det_gts[0].shape[1]
        flag_det_gts = self.get_flatten_boxes(flag_det_gts, num_classes, flag=True)
        det_gts = self.get_flatten_boxes(det_gts, num_classes)
        det_gts_decode = self.get_flatten_boxes(det_gts_decode, num_classes)
        det_preds = self.get_flatten_boxes(det_preds, num_classes)
        det_preds_decode = self.get_flatten_boxes(det_preds_decode, num_classes)
        det_gts_dict = {k: det_gts[k, flag_det_gts[k]] for k in range(num_classes)}
        det_preds_dict = {k: det_preds[k, flag_det_gts[k]] for k in range(num_classes)}
        det_gts_decode_dict = {k: det_gts_decode[k, flag_det_gts[k]] for k in range(num_classes)}
        det_preds_decode_dict = {k: det_preds_decode[k, flag_det_gts[k]] for k in range(num_classes)}
        return det_gts_dict, det_preds_dict, det_gts_decode_dict, det_preds_decode_dict

    def get_topiou_predictions(self, threshold, seg_preds, targets, image_shape, flag_det_gts, det_gts, det_preds, det_gts_decode, det_preds_decode, use_seg=False, sigma=2.0):
        seg_preds = [seg_preds[k] for k in self.det_level_in_seg]
        expected_ious = self.get_expected_ious(seg_preds, targets, image_shape)
        num_classes = seg_preds[0].shape[1]
        det_gts_dict, det_preds_dict = {k:[]for k in range(num_classes)}, {k:[]for k in range(num_classes)}
        det_gts_decode_dict, det_preds_decode_dict = {k:[]for k in range(num_classes)}, {k:[]for k in range(num_classes)}
        for k in range(len(seg_preds)):
            flag_det_gt = flag_det_gts[k].permute(1,0,3,4,2).reshape(num_classes, -1) #Cx(BMN)
            seg_pred = seg_preds[k].unsqueeze(2).permute(1,0,3,4,2).reshape(num_classes, -1) #Cx(BMN)
            iou = expected_ious[k].unsqueeze(2).permute(1,0,3,4,2).reshape(num_classes, -1) #Cx(BMN)
            if use_seg:
                iou = torch.pow(iou, (sigma-seg_pred.detach())/sigma)
            det_gt = det_gts[k].permute(1,0,3,4,2).reshape(num_classes, -1, 4) #Cx(BMN)x4
            det_pred = det_preds[k].permute(1,0,3,4,2).reshape(num_classes, -1, 4) #Cx(BMN)x4
            det_gt_decode = det_gts_decode[k].permute(1,0,3,4,2).reshape(num_classes, -1, 4) #Cx(BMN)x4
            det_pred_decode = det_preds_decode[k].permute(1,0,3,4,2).reshape(num_classes, -1, 4) #Cx(BMN)x4
            for c in range(num_classes):
                flag = flag_det_gt[c] & (iou[c] > threshold)
                det_gts_dict[c].append(det_gt[c,flag])
                det_preds_dict[c].append(det_pred[c,flag])
                det_gts_decode_dict[c].append(det_gt_decode[c,flag])
                det_preds_decode_dict[c].append(det_pred_decode[c,flag])
                # print(flag.sum(), det_gt[c,flag].shape, det_gt_decode[c,flag].shape)
        det_gts_dict = {key: torch.cat(values, dim=0) for key, values in det_gts_dict.items()} 
        det_preds_dict = {key: torch.cat(values, dim=0) for key, values in det_preds_dict.items()} 
        det_gts_decode_dict = {key: torch.cat(values, dim=0) for key, values in det_gts_decode_dict.items()} 
        det_preds_decode_dict = {key: torch.cat(values, dim=0) for key, values in det_preds_decode_dict.items()} 
        return det_gts_dict, det_preds_dict, det_gts_decode_dict, det_preds_decode_dict

    def compute_detection_loss(self, seg_preds, flag_det_gts, det_gts, det_preds, det_gts_decode, det_preds_decode, targets=None, image_shape=None):
        det_losses = {}
        num_classes = flag_det_gts[0].shape[1]
        losses_func = self.losses_func['detection_losses']
        loss_weights = self.loss_weights['detection_losses']

        # for CDR calculation
        smooth = 1e-10
        _, _, det_gt_dec, det_pred_dec = self.get_topk_predictions(1, seg_preds, flag_det_gts, det_gts, det_preds, det_gts_decode, det_preds_decode)
        # cdr_gt = (det_gt_dec[0][:,2:] - det_gt_dec[0][:,:2]) / (det_gt_dec[1][:,2:] - det_gt_dec[1][:,:2] + smooth)
        # cdr_pd = (det_pred_dec[0][:,2:] - det_pred_dec[0][:,:2]) / (det_pred_dec[1][:,2:] - det_pred_dec[1][:,:2] + smooth)
        cup_gt = det_gt_dec[0][:,2:] - det_gt_dec[0][:,:2]
        disc_gt = det_gt_dec[1][:,2:] - det_gt_dec[1][:,:2]
        cup_pd = det_pred_dec[0][:,2:] - det_pred_dec[0][:,:2]
        disc_pd = det_pred_dec[1][:,2:] - det_pred_dec[1][:,:2]
        kwargs_opt = {'cup_gt': cup_gt, 'disc_gt': disc_gt, 'cup_pd': cup_pd, 'disc_pd': disc_pd}
        for loss_func, loss_w in zip(losses_func, loss_weights):
            if 'CDR' not in type(loss_func).__name__:
                continue
            loss_keys = loss_func.__call__.__code__.co_varnames
            loss_params = {key:kwargs_opt[key] for key in kwargs_opt.keys() if key in loss_keys}
            loss_v = loss_func(**loss_params)*loss_w

            loss_v = {type(loss_func).__name__: loss_v}
            det_losses.update(loss_v)

        if self.detection_sample_selection['method'] == 'all':
            det_gts, det_preds, det_gts_decode, det_preds_decode = \
                self.get_all_predictions(flag_det_gts, det_gts, det_preds, det_gts_decode, det_preds_decode)
        elif self.detection_sample_selection['method'] == 'iou':
            det_gts, det_preds, det_gts_decode, det_preds_decode = \
                self.get_topiou_predictions(self.detection_sample_selection['iou_th'], 
                                            seg_preds, targets, image_shape, flag_det_gts, 
                                            det_gts, det_preds, det_gts_decode, det_preds_decode,
                                            self.detection_sample_selection['use_seg'],
                                            self.detection_sample_selection['sigma'])
        elif self.detection_sample_selection['method'] == 'topk':
            det_gts, det_preds, det_gts_decode, det_preds_decode = \
                self.get_topk_predictions(self.detection_sample_selection['topk'], seg_preds, flag_det_gts, det_gts, det_preds, det_gts_decode, det_preds_decode)
                
        for k in range(num_classes):
            kwargs_opt = {'ypred_reg': det_preds[k], 'ytrue_reg': det_gts[k],
                'ypred_decode_reg': det_preds_decode[k], 'ytrue_decode_reg': det_gts_decode[k]}
            
            for loss_func, loss_w in zip(losses_func, loss_weights):
                if 'CDR' in type(loss_func).__name__:
                    continue
                loss_keys = loss_func.__call__.__code__.co_varnames
                loss_params = {key:kwargs_opt[key] for key in kwargs_opt.keys() if key in loss_keys}
                loss_v = loss_func(**loss_params)*loss_w

                loss_v = {type(loss_func).__name__+'/'+str(k): loss_v}
                det_losses.update(loss_v)
        return det_losses

    def postprocess_detections(self, pred_segs, pred_boxes, image_shapes):
        device = pred_segs[0].device
        num_classes = pred_segs[0].shape[1]
        batch_size = pred_segs[0].shape[0]
        
        pred_segs = torch.cat([v.reshape(batch_size, num_classes, -1) \
                for v in pred_segs], dim=2) # CxBx(HxW)
        pred_boxes = torch.cat([v.permute(0,1,3,4,2).reshape(
            batch_size, num_classes, -1, 4) for v in pred_boxes], dim=2)# CxBx(HxW)x4
        
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        for bs in range(pred_boxes.shape[0]):
            boxes, scores, image_shape = pred_boxes[bs], pred_segs[bs], image_shapes[bs]
            image_shape = torch.tensor(image_shape, dtype=torch.float32, device=device)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_all_boxes = []
            image_all_scores = []
            image_all_labels = []

            for n_cls in range(num_classes):
                # remove low scoring boxes
                inds = torch.where(scores[n_cls] > self.nms_score_threshold)[0]
                if len(inds)==0:
                    continue
                box = boxes[n_cls, inds, :]
                sc  = scores[n_cls, inds]

                # remove empty boxes
                keep = box_ops.remove_small_boxes(box, min_size=1e-2)
                box, sc = box[keep], sc[keep]

                if self.post_obj:
                    # keep the highest prob.
                    ind_sort = torch.argsort(sc, descending=True)
                    keep = ind_sort[:self.detections_per_class]
                else:
                    # non-maximum suppression
                    keep = box_ops.nms(box, sc, self.nms_iou_threshold)
                    keep = keep[:self.detections_per_class]
                box, sc = box[keep], sc[keep]
                labels = torch.tensor([n_cls]*len(sc), dtype=torch.int, device=device)

                image_all_boxes.append(box)
                image_all_scores.append(sc)
                image_all_labels.append(labels)
            if len(image_all_boxes)>0:
                image_all_boxes = torch.cat(image_all_boxes, dim=0)
                image_all_scores = torch.cat(image_all_scores, dim=0)
                image_all_labels = torch.cat(image_all_labels, dim=0)
            else:
                image_all_boxes = torch.empty((0,4), dtype=torch.float32, device=device)
                image_all_scores = torch.empty((0,1), dtype=torch.float32, device=device)
                image_all_labels = torch.empty((0,1), dtype=torch.float32, device=device)
            result.append({
                        "boxes":  image_all_boxes,
                        "labels": image_all_labels,
                        "scores": image_all_scores,
                    })

        return result
    
    def forward(self, images, targets=None, dense_results=False):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.batch_image(images, targets)
        if torch.isnan(images.tensors).sum()>0:
            print('image is nan ..............')
        if torch.isinf(images.tensors).sum()>0:
            print('image is inf ..............')  

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        seg_preds, det_preds = self.model(images.tensors)
        # print('--------------image and seg outputs: ')
        # print(">>> seg_preds", [v.shape for v in seg_preds])
        # print(">>> det_preds", [v.shape for v in det_preds])

        image_shape = images.tensors.shape[-2:]
        det_gts, flag_det_gts = self.get_bbox_regression_targets(det_preds, targets, image_shape)
        det_preds_decode = self.bbox_decoder(det_preds, image_shape)
        det_gts_decode = self.bbox_decoder(det_gts, image_shape)
        # expected_ious = self.get_expected_ious([seg_preds[k] for k in self.det_level_in_seg], targets, image_shape)
        det_losses = self.compute_detection_loss(seg_preds, flag_det_gts, det_gts, det_preds, det_gts_decode, det_preds_decode, targets, image_shape)

        # ## plot det_gts
        # import matplotlib.pylab as plt
        # for ipx, ious in enumerate(expected_ious):
        #     for bs in range(ious.shape[0]):
        #         target = targets[bs]
        #         image = images.tensors[bs,0].numpy()
        #         iou = ious[bs]>0.6
        #         print(target["masks"].shape, iou[bs].shape)
        #         plt.figure()
        #         plt.subplot(2,2,1)
        #         plt.imshow(image)
        #         plt.subplot(2,2,2)
        #         plt.imshow(target["masks"].sum(dim=0).numpy())
        #         plt.subplot(2,2,3)
        #         plt.imshow(iou[0].numpy()+target["masks"][0].numpy())
        #         plt.subplot(2,2,4)
        #         plt.imshow(iou[1].numpy()+target["masks"][1].numpy())
        #         plt.savefig(f"tmp_{bs}_iou.jpg")
        # import sys
        # sys.exit()

        ## calculate losses
        assert targets is not None
        if self.softmax:
            losses = self.softmax_compute_seg_loss(seg_preds, targets, images.tensors.shape)
        else:
            losses = self.sigmoid_compute_seg_loss(seg_preds, targets, images.tensors.shape)
        losses.update(det_losses)

        if dense_results:
            return losses, [det_preds_decode, seg_preds]
        else:
            all_boxes = self.postprocess_detections([seg_preds[k] for k in self.det_level_in_seg], 
                        det_preds_decode, images.image_sizes)
            all_boxes = self.batch_image.postprocess(all_boxes, images.image_sizes, original_image_sizes)
            return losses, [all_boxes, seg_preds]

    
