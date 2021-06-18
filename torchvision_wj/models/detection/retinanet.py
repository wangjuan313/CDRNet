from collections import OrderedDict

import torch
from torch import nn
from torchvision_wj.models.batch_image import BatchImage
from .anchors import AnchorGenerator, BoxCoder, anchor_targets_bbox
from typing import Union
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor
import math
from torchvision.ops import boxes as box_ops

__all__ = ["Retinanet", "RetinanetHead"]

class Retinanet(nn.Module):

    def __init__(self, backbone, losses, loss_weights, heads=None, num_classes=None, 
                 # data transform parameters
                 # transform=GeneralizedRCNNTransform, 
                 # min_size=800, max_size=1333,
                 # image_mean=None, image_std=None,
                 batch_image=BatchImage, size_divisible=32,
                 # head parameters
                 num_conv=4, feature_size=256, prior=0.01,
                 # anchor parameters
                 anchor_sizes=((32,), (64,), (128,), (256,), (512,)), 
                 anchor_aspect_ratios=((0.5, 1.0, 2.0),) * 5, 
                 anchor_scales=((2**0, 2**(1/3), 2**(2/3)),) * 5,
                 # Box parameters
                 # box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_high_iou_thresh=0.5, box_low_iou_thresh=0.4,
                 box_stat_mean=[0,0,0,0], box_stat_std=[0.2,0.2,0.2,0.2],
                 # NMS parameters
                 nms_score_threshold=0.005, nms_iou_threshold=0.05,
                 detections_per_class=10
                 ):

        super(Retinanet, self).__init__()

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        self.backbone       = backbone
        out_channels        = backbone.out_channels
        self.losses_func    = losses
        self.loss_weights   = loss_weights
        self.num_classes    = num_classes

        self.anchor_generator = AnchorGenerator(anchor_sizes, anchor_aspect_ratios, anchor_scales)
        num_anchors = self.anchor_generator.num_anchors_per_location()[0]
        self.num_anchors = num_anchors

        if heads is None:
            self.heads = RetinanetHead(
                out_channels, num_anchors, num_conv=num_conv,
                feature_size=feature_size, num_classes=num_classes, num_fpn=len(anchor_sizes), prior=0.01)
        else:
            self.heads = heads

        self.batch_image = batch_image(size_divisible)

        self.box_coder = BoxCoder(mean=box_stat_mean, std=box_stat_std)

        self.box_low_iou_thresh  = box_low_iou_thresh
        self.box_high_iou_thresh = box_high_iou_thresh

        self.nms_score_threshold  = nms_score_threshold
        self.nms_iou_threshold    = nms_iou_threshold
        self.detections_per_class = detections_per_class
        # self.loss_params          = loss_params

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def compute_loss(self, pred_cls, pred_bbox_deltas, labels_targets, regression_targets, anchor_state):
        pred_cls           = pred_cls.flatten(0, -2)
        pred_bbox_deltas   = pred_bbox_deltas.flatten(0, -2)
        labels_targets     = labels_targets.flatten(0, -2)
        regression_targets = regression_targets.flatten(0, -2)
        anchor_state       = anchor_state.flatten()

        index_cls          = anchor_state!=-1
        index_bbox         = anchor_state==1

        kwargs_opt = {'ypred':pred_cls[index_cls,:], 'ytrue':labels_targets[index_cls,:], 
                      'ypred_reg':pred_bbox_deltas[index_bbox,:], 'ytrue_reg':regression_targets[index_bbox,:]}
        det_losses = {}
        for loss_func, loss_w in zip(self.losses_func,self.loss_weights):
            loss_keys = loss_func.__call__.__code__.co_varnames
            loss_params = {key:kwargs_opt[key] for key in kwargs_opt.keys() if key in loss_keys}
            loss_v = loss_func(**loss_params)*loss_w

            loss_v = {type(loss_func).__name__:loss_v}
            det_losses.update(loss_v)
        # print(det_losses)
        return det_losses

    def postprocess_detections(self,
                               pred_cls,       # type: Tensor
                               pred_bbox_deltas,  # type: Tensor
                               anchors,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = pred_cls.device
        num_classes = pred_cls.shape[-1]
        pred_boxes = self.box_coder.decode(torch.stack(anchors, dim=0), pred_bbox_deltas)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        for bs in range(pred_boxes.shape[0]):
            boxes, scores, image_shape = pred_boxes[bs], pred_cls[bs], image_shapes[bs]
            image_shape = torch.tensor(image_shape, dtype=torch.float32, device=device)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_all_boxes = []
            image_all_scores = []
            image_all_labels = []

            for n_cls in range(num_classes):
                # remove low scoring boxes
                inds = torch.where(scores[:,n_cls] > self.nms_score_threshold)[0]
                if len(inds)==0:
                    # print(scores[:,n_cls].max())
                    # print('no detections for label = {}'.format(n_cls))
                    continue
                box = boxes[inds, :]
                sc  = scores[inds, n_cls]
                # print('candidata box {:d}'.format(len(inds)))

                # remove empty boxes
                keep = box_ops.remove_small_boxes(box, min_size=1e-2)
                box, sc = box[keep], sc[keep]

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
        # print("original image size: ", original_image_sizes)

        # images, targets = self.transform(images, targets)
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

        features = self.backbone(images.tensors)
        features = list(features.values())
        # print('--------------image and feature maps: ')
        # print(images.tensors.shape)
        # print([f.shape for f in features])
        pred_cls_org, pred_bbox_deltas_org = self.heads(features)
        # print('pred shape'.center(50,'-'))
        # print([pred.shape for pred in pred_cls])
        pred_cls, pred_bbox_deltas = concat_box_prediction_layers(pred_cls_org,pred_bbox_deltas_org)
        anchors = self.anchor_generator(images, features)
        # print("<<< ", pred_cls.shape, pred_bbox_deltas.shape)

        ## calculate losses
        assert targets is not None
        regression_targets, labels_targets, anchor_state = \
                                    anchor_targets_bbox(anchors, targets, self.num_classes, images.image_sizes, 
                                    negative_overlap=self.box_low_iou_thresh,
                                    positive_overlap=self.box_high_iou_thresh)
        regression_targets = self.box_coder.encode(torch.stack(anchors, dim=0), regression_targets)
        losses = self.compute_loss(pred_cls, pred_bbox_deltas, 
                                   labels_targets, regression_targets, anchor_state)

        ## calculate detections
        if dense_results:
            feat_size = [[pred.shape[2],pred.shape[3]] for pred in pred_cls_org]
            split_size = [self.num_anchors*pred.shape[2]*pred.shape[3] for pred in pred_cls_org]
            pred_boxes = self.box_coder.decode(torch.stack(anchors, dim=0), pred_bbox_deltas)
            pred_boxes = torch.split(pred_boxes, split_size, dim=1)
            pred_boxes = [pred.reshape(pred.shape[0],sz[0],sz[1],self.num_anchors,pred.shape[2]) \
                          for sz,pred in zip(feat_size,pred_boxes)]
            pred_boxes = [pred.permute(0,3,4,1,2) for pred in pred_boxes]
            pred_cls = [pred.reshape(pred.shape[0],self.num_anchors,self.num_classes,pred.shape[2],pred.shape[3]) \
                        for pred in pred_cls_org]
            # print("<<< pred_boxes", [v.shape for v in pred_boxes])
            # print("<<< pred_cls", [v.shape for v in pred_cls])
            all_boxes = [pred_cls, pred_boxes]
        else:
            all_boxes = self.postprocess_detections(\
                                            pred_cls, pred_bbox_deltas, anchors, images.image_sizes)
            all_boxes = self.batch_image.postprocess(all_boxes, images.image_sizes, original_image_sizes)
        return losses, all_boxes

class RetinanetHeadSingle(nn.Module):
    def __init__(self, in_channels, num_anchors, num_conv=4, feature_size=256, num_classes=81, prior=0.01):
        super(RetinanetHeadSingle, self).__init__()
        self.in_channels  = in_channels
        self.feature_size = feature_size
        self.num_conv     = num_conv
        self.cls_feat  = self._make_layers()
        self.bbox_feat = self._make_layers()
        self.cls_logits = nn.Sequential(OrderedDict([
                                      ('cls',    nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)),
                                      ('sigmod', nn.Sigmoid())
                                    ]))
        self.bbox_pred = nn.Conv2d(feature_size, num_anchors*4, kernel_size=3, padding=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        bias_value = -(math.log((1 - prior) / prior))
        self.cls_logits.cls.bias.data.fill_(bias_value)

    def _make_layers(self):
        in_channels  = self.in_channels
        feature_size = self.feature_size
        num_conv     = self.num_conv
        layers = OrderedDict()
        layers['conv1'] = nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1)
        layers['relu1'] = nn.ReLU()
        for k in range(1,num_conv):
            layers['conv'+str(k+1)] = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
            layers['relu'+str(k+1)] = nn.ReLU()
        return nn.Sequential(layers)

    def forward(self, x):
        cls_feat  = self.cls_feat(x)
        bbox_feat = self.bbox_feat(x)
        logits    = self.cls_logits(cls_feat)
        bbox_reg  = self.bbox_pred(bbox_feat)
        return logits, bbox_reg

class RetinanetHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_conv=4, feature_size=256, num_classes=81, num_fpn=4, prior=0.01):
        super(RetinanetHead, self).__init__()
        
        self.heads_list = nn.ModuleList()
        for k in range(num_fpn):
            single_head = RetinanetHeadSingle(in_channels, num_anchors, num_conv, feature_size, num_classes, prior)
            self.heads_list.append(single_head)
        nb_params = []
        for model in self.heads_list:
            nb_params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('# trainable parameters in heads: {}'.format(nb_params))

    def forward(self, inputs):
        assert len(self.heads_list)==len(inputs)
        logits, bbox_reg = [], []
        for x,heads in zip(inputs,self.heads_list):
            v_cls,v_reg = heads(x)
            logits.append(v_cls)
            bbox_reg.append(v_reg)
        return logits, bbox_reg

def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int) -> Tensor
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2).contiguous()
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1)#.flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1)#.reshape(-1, 4)
    return box_cls, box_regression