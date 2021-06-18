
import torch
import torch.nn as nn
from torchvision.ops import boxes
from torch.jit.annotations import List, Optional, Dict, Tuple
import math

class AnchorGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Arguments:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
        scales = ((2**0,2**(1/3),2**(2/3)),)
    ):
        super(AnchorGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.cell_anchors = None
        self._cache = {}

    # TODO: https://github.com/pytorch/pytorch/issues/26792
    # For every (aspect_ratios, scales) combination, output a zero-centered anchor with those values.
    # (scales, aspect_ratios) are usually an element of zip(self.scales, self.aspect_ratios)
    # This method assumes aspect ratio = height / width for an anchor.
    def generate_anchors(self, base_size, aspect_ratios, scales, dtype=torch.float32, device="cpu"):
        # type: (List[int], List[float], int, Device) -> Tensor  # noqa: F821
        base_size = torch.as_tensor(base_size, dtype=dtype, device=device)
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        
        ws = base_size*(w_ratios[:, None] * scales[None, :]).view(-1)
        hs = base_size*(h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors

    def set_cell_anchors(self, dtype, device):
        # type: (int, Device) -> None  # noqa: F821
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            # suppose that all anchors have the same device
            # which is a valid assumption in the current state of the codebase
            if cell_anchors[0].device == device:
                return

        cell_anchors = [
            self.generate_anchors(
                sizes,
                aspect_ratios,
                scales,
                dtype,
                device
            )
            for sizes, aspect_ratios, scales in zip(self.sizes, self.aspect_ratios, self.scales)
        ]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.scales, self.aspect_ratios)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None
        assert len(grid_sizes) == len(strides) == len(cell_anchors)
        

        for size, stride, base_anchors in zip(
            grid_sizes, strides, cell_anchors
        ):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = (torch.arange(
                0, grid_width, dtype=torch.float32, device=device
            )+0.5) * stride_width
            shifts_y = (torch.arange(
                0, grid_height, dtype=torch.float32, device=device
            )+0.5) * stride_height
            
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor]) -> List[Tensor]
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        # Clear the cache in case that memory leaks.
        self._cache.clear()
        return anchors
    
def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    # The Mean and std are calculated from COCO dataset.
    # Bounding box normalization was firstly introduced in the Fast R-CNN paper.
    # See https://github.com/fizyr/keras-retinanet/issues/1273#issuecomment-585828825  for more details
    if mean is None:
        mean = torch.tensor([0, 0, 0, 0])
    if std is None:
        std = torch.tensor([0.2, 0.2, 0.2, 0.2])

    anchor_widths  = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    # According to the information provided by a keras-retinanet author, they got marginally better results using
    # the following way of bounding box parametrization.
    # See https://github.com/fizyr/keras-retinanet/issues/1273#issuecomment-585828825 for more details
    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

    targets = torch.cat((targets_dx1.reshape(-1,1), targets_dy1.reshape(-1,1), 
                        targets_dx2.reshape(-1,1), targets_dy2.reshape(-1,1)), dim=1)

    targets = (targets - mean) / std

    return targets

def bbox_transform_batch(anchors_batch, gt_boxes_batch, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    # The Mean and std are calculated from COCO dataset.
    # Bounding box normalization was firstly introduced in the Fast R-CNN paper.
    # See https://github.com/fizyr/keras-retinanet/issues/1273#issuecomment-585828825  for more details
    if mean is None:
        mean = torch.tensor([0, 0, 0, 0])
    if std is None:
        std = torch.tensor([0.2, 0.2, 0.2, 0.2])

    anchor_widths  = anchors_batch[:, :, 2] - anchors_batch[:, :, 0]
    anchor_heights = anchors_batch[:, :, 3] - anchors_batch[:, :, 1]

    # According to the information provided by a keras-retinanet author, they got marginally better results using
    # the following way of bounding box parametrization.
    # See https://github.com/fizyr/keras-retinanet/issues/1273#issuecomment-585828825 for more details
    targets_dx1 = (gt_boxes_batch[:, :, 0] - anchors_batch[:, :, 0]) / anchor_widths
    targets_dy1 = (gt_boxes_batch[:, :, 1] - anchors_batch[:, :, 1]) / anchor_heights
    targets_dx2 = (gt_boxes_batch[:, :, 2] - anchors_batch[:, :, 2]) / anchor_widths
    targets_dy2 = (gt_boxes_batch[:, :, 3] - anchors_batch[:, :, 3]) / anchor_heights

    targets = torch.stack([targets_dx1, targets_dy1, 
                           targets_dx2, targets_dy2], dim=2)

    targets = (targets - mean[None,None,:]) / std[None,None,:]

    return targets

def bbox_transform_inv_batch(anchors_batch, deltas_batch, mean=None, std=None):
    """ Applies deltas (usually regression results) to boxes (usually anchors).

    Before applying the deltas to the boxes, the normalization that was previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the boxes.

    Args
        boxes : np.array of shape (B, N, 4), where B is the batch size, N the number of boxes and 4 values for (x1, y1, x2, y2).
        deltas: np.array of same shape as boxes. These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean  : The mean value used when computing deltas (defaults to [0, 0, 0, 0]).
        std   : The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2]).

    Returns
        A np.array of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """
    width  = anchors_batch[:, :, 2] - anchors_batch[:, :, 0]
    height = anchors_batch[:, :, 3] - anchors_batch[:, :, 1]

    x1 = anchors_batch[:, :, 0] + (deltas_batch[:, :, 0] * std[0] + mean[0]) * width
    y1 = anchors_batch[:, :, 1] + (deltas_batch[:, :, 1] * std[1] + mean[1]) * height
    x2 = anchors_batch[:, :, 2] + (deltas_batch[:, :, 2] * std[2] + mean[2]) * width
    y2 = anchors_batch[:, :, 3] + (deltas_batch[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = torch.stack([x1, y1, x2, y2], axis=2)
    return pred_boxes

class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, mean=[0,0,0,0], std=[0.2,0.2,0.2,0.2], bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.mean = mean
        self.std  = std
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, anchors_group, gt_boxes_group):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        dtype = anchors_group.dtype
        device = anchors_group.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
        std  = torch.as_tensor(self.std, dtype=dtype, device=device)
        targets_group = bbox_transform_batch(anchors_group, gt_boxes_group, mean, std)
        return targets_group

    def encode_single(self, anchors, gt_boxes):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = anchors.dtype
        device = anchors.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
        std  = torch.as_tensor(self.std, dtype=dtype, device=device)
        targets = bbox_transform(anchors, gt_boxes, mean, std)
        return targets

    def decode(self, anchors_group, deltas_group):
        # type: (Tensor, Tensor) -> Tensor
        dtype = deltas_group.dtype
        device = deltas_group.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
        std  = torch.as_tensor(self.std, dtype=dtype, device=device)
        pred_boxes = bbox_transform_inv_batch(anchors_group, deltas_group, mean, std)
        return pred_boxes

def clip_boxes(boxes, image_sizes):
    boxes_clip = []
    for n in range(boxes.shape[0]):
        b = boxes[n]
        height, width = image_sizes[0]

        b[:, 0] = torch.clamp(b[:, 0], min=0)
        b[:, 1] = torch.clamp(b[:, 1], min=0)
        b[:, 2] = torch.clamp(b[:, 2], max=width)
        b[:, 3] = torch.clamp(b[:, 3], max=height)
        boxes_clip.append(b) 
    boxes_clip = torch.stack(boxes_clip, dim=0)
    return boxes_clip

def anchor_targets_bbox(
    anchors_group,
    annotations_group,
    num_classes,
    image_size_group=None,
    negative_overlap=0.4,
    positive_overlap=0.5
):
    """ Generate anchor targets for bbox detection.

    Args
        anchors_group: List of anchors with each anchor shape (N, 4) for (x1, y1, x2, y2).
        annotations_group: List of annotation dictionaries with each annotation containing 'labels' and 'boxes' of an image.
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used to mark the relevant part of the image.
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        labels_batch: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
                      where N is the number of anchors for an image and the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: batch that contains bounding-box regression targets for an image & anchor states (np.array of shape (batch_size, N, 4 + 1),
                      where N is the number of anchors for an image, the first 4 columns define regression targets for (x1, y1, x2, y2) and the
                      last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """

    assert(len(annotations_group) > 0), "No data received to compute anchor targets."
    for annotations in annotations_group:
        assert('boxes' in annotations), "Annotations should contain bboxes."
        assert('labels' in annotations), "Annotations should contain labels."

    batch_size = len(annotations_group)
    device = anchors_group[0].device
    regression_batch  = torch.zeros((batch_size, anchors_group[0].shape[0], 4), dtype=torch.float32, device=device)
    labels_batch      = torch.zeros((batch_size, anchors_group[0].shape[0], num_classes), dtype=torch.float32, device=device)
    anchor_state      = torch.zeros((batch_size, anchors_group[0].shape[0]), dtype=torch.float32, device=device)

    # compute labels and regression targets
    for index, (anchors,annotations,image_size) in enumerate(zip(anchors_group,annotations_group,image_size_group)):
        bboxes = annotations['boxes']

        if bboxes.shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(anchors, bboxes, negative_overlap, positive_overlap)

            # labels_batch[index, ignore_indices, -1]       = -1
            # labels_batch[index, positive_indices, -1]     = 1

            # regression_batch[index, ignore_indices, -1]   = -1
            # regression_batch[index, positive_indices, -1] = 1

            anchor_state[index, ignore_indices]   = -1
            anchor_state[index, positive_indices] = 1

            # compute target class labels
            labels_batch[index, positive_indices, annotations['labels'][argmax_overlaps_inds[positive_indices]]] = 1

            # regression_batch[index, :, :-1] = bbox_transform(anchors, bboxes[argmax_overlaps_inds, :])
            regression_batch[index] = bboxes[argmax_overlaps_inds, :]

        # ignore annotations outside of image
        if image_size:
            image_size = torch.tensor(image_size, dtype=torch.float32, device=device)
            anchors_centers = torch.cat([(anchors[:, 0]+anchors[:, 2]).reshape(-1,1)/2, (anchors[:, 1]+anchors[:, 3]).reshape(-1,1)/2],dim=1)
            indices = torch.logical_or(anchors_centers[:,0]>=image_size[1], anchors_centers[:,1]>=image_size[0])

            # labels_batch[index, indices, -1]     = -1
            # regression_batch[index, indices, -1] = -1
            anchor_state[index, indices]   = -1

    return regression_batch, labels_batch, anchor_state

def compute_gt_annotations(
    anchors,
    annotations,
    negative_overlap=0.4,
    positive_overlap=0.5):
    """ Obtain indices of gt annotations with the greatest overlap.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (N, 5) for (x1, y1, x2, y2, label).
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        positive_indices: indices of positive anchors
        ignore_indices: indices of ignored anchors
        argmax_overlaps_inds: ordered overlaps indices
    """

    overlaps = boxes.box_iou(anchors, annotations)
    max_overlaps, argmax_overlaps_inds = torch.max(overlaps, dim=1)

    # assign "dont care" labels
    positive_indices = max_overlaps >= positive_overlap
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_inds



# if __name__=='__main__':
    # import numpy as np
    # from ref_anchors import generate_anchors, anchors_for_shape, shift
    # sizes=((64,),(32,),(16,),(8,),(4,))
    # aspect_ratios=((0.5, 1.0, 2.0),)
    # scales=((2**0,2**(1/3),2**(2/3)),)
    # generator = AnchorGenerator(sizes=sizes,
    #     aspect_ratios=aspect_ratios*len(sizes),
    #     scales=scales*len(sizes)
    #     )
    
    # anchors = generator.generate_anchors(base_size=((16),),
    #                                      scales=((2**0,2**(1/3),2**(2/3)),),
    #                                      aspect_ratios=((0.5, 1.0, 2.0)),)
    
    # ref = generate_anchors(base_size=16, ratios=(0.5, 1.0, 2.0), scales=(2**0,2**(1/3),2**(2/3)))
    # np.testing.assert_almost_equal(anchors.numpy(),ref.numpy())
    # ratios=(0.5, 1.0, 2.0)
    # ref_scales=(2**0,2**(1/3),2**(2/3))
    # sizes = (64,32,16,8,4)
    # ref_anchors = []
    # for s in sizes:
    #     ref = generate_anchors(base_size=s, ratios=ratios, scales=ref_scales)
    #     ref_anchors.append(ref)
        
    # generator.set_cell_anchors(torch.float32,'cpu')
    # for a1,a2 in zip(ref_anchors,generator.cell_anchors):
    #     print(a1.shape,a2.shape)
    #     np.testing.assert_almost_equal(a1.numpy(),a2.numpy())


    # sizes=((32,))
    # generator = AnchorGenerator(sizes=sizes,
    #     aspect_ratios=aspect_ratios*len(sizes),
    #     scales=scales*len(sizes)
    #     )
    # grid_sizes  = ((8,12),)
    # strides = ((64,64),)
    # anchors = generate_anchors(base_size=32, ratios=ratios, scales=ref_scales)
    # shifted_anchors = shift(grid_sizes[0], strides[0][0], anchors)    
    # generator.set_cell_anchors(torch.float32,'cpu')
    # np.testing.assert_almost_equal(anchors.numpy(),generator.cell_anchors[0].numpy())
    # anchors_all = generator.cached_grid_anchors(grid_sizes, strides)
    # np.testing.assert_almost_equal(shifted_anchors.numpy(),anchors_all[0].numpy())
    
    
    # sizes=((32,),(64,),(128,),(256,),(512,))
    # generator = AnchorGenerator(sizes=sizes,
    #     aspect_ratios=aspect_ratios*len(sizes),
    #     scales=scales*len(sizes)
    #     )
    # image_shape = (512,512)
    # grid_sizes  = ((64,64),(32,32),(16,16),(8,8),(4,4))
    # strides = ((8,8), (16,16), (32,32), (64,64), (128,128))
    # ref_all = anchors_for_shape(image_shape)
    # generator.set_cell_anchors(torch.float32,'cpu')
    # anchors_all = generator.cached_grid_anchors(grid_sizes, strides)
    # for a1,a2 in zip(ref_all,anchors_all):
    #     print(a1.shape,a2.shape)
    #     d = torch.abs(a1 - a2)
    #     print(d.max(),d.min())
    #     np.testing.assert_almost_equal(a1.numpy(),a2.numpy(),decimal=4)
    
    # shift_x = torch.arange(0, 5) 
    # shift_y = torch.arange(0, 2) 
    # shift_x1, shift_y1 = torch.meshgrid([shift_x, shift_y])
    # shift_y2, shift_x2 = torch.meshgrid([shift_y, shift_x])
    # shift_y3, shift_x3 = torch.meshgrid(shift_y, shift_x)
    # v1 = torch.stack((shift_x1.reshape(-1), shift_y1.reshape(-1)), dim=1)
    # v2 = torch.stack((shift_x2.reshape(-1), shift_y2.reshape(-1)), dim=1)
    # v3 = torch.stack((shift_x3.reshape(-1), shift_y3.reshape(-1)), dim=1)