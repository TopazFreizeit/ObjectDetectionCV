import torch.nn as nn
import torch
import math
import torchvision


def get_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])

    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    intersection_area = ((x_right - x_left).clamp(min=0) *
                         (y_bottom - y_top).clamp(min=0))
    union = area1[:, None] + area2 - intersection_area
    iou = intersection_area / union
    return iou


def boxes_to_transformation_targets(ground_truth_boxes,
                                    default_boxes,
                                    weights=(10., 10., 5., 5.)):
    widths = default_boxes[:, 2] - default_boxes[:, 0]
    heights = default_boxes[:, 3] - default_boxes[:, 1]
    center_x = default_boxes[:, 0] + 0.5 * widths
    center_y = default_boxes[:, 1] + 0.5 * heights

    gt_widths = (ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0])
    gt_heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
    gt_center_x = ground_truth_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = ground_truth_boxes[:, 1] + 0.5 * gt_heights

    targets_dx = weights[0] * (gt_center_x - center_x) / widths
    targets_dy = weights[1] * (gt_center_y - center_y) / heights
    targets_dw = weights[2] * torch.log(gt_widths / widths)
    targets_dh = weights[3] * torch.log(gt_heights / heights)
    regression_targets = torch.stack((targets_dx,
                                      targets_dy,
                                      targets_dw,
                                      targets_dh), dim=1)
    return regression_targets


def apply_regression_pred_to_default_boxes(box_transform_pred,
                                           default_boxes,
                                           weights=(10., 10., 5., 5.)):
    w = default_boxes[:, 2] - default_boxes[:, 0]
    h = default_boxes[:, 3] - default_boxes[:, 1]
    center_x = default_boxes[:, 0] + 0.5 * w
    center_y = default_boxes[:, 1] + 0.5 * h

    dx = box_transform_pred[..., 0] / weights[0]
    dy = box_transform_pred[..., 1] / weights[1]
    dw = box_transform_pred[..., 2] / weights[2]
    dh = box_transform_pred[..., 3] / weights[3]

    pred_center_x = dx * w + center_x
    pred_center_y = dy * h + center_y
    pred_w = torch.exp(dw) * w
    pred_h = torch.exp(dh) * h

    pred_box_x1 = pred_center_x - 0.5 * pred_w
    pred_box_y1 = pred_center_y - 0.5 * pred_h
    pred_box_x2 = pred_center_x + 0.5 * pred_w
    pred_box_y2 = pred_center_y + 0.5 * pred_h

    pred_boxes = torch.stack((
        pred_box_x1,
        pred_box_y1,
        pred_box_x2,
        pred_box_y2),
        dim=-1)
    return pred_boxes


def generate_default_boxes(feat_sizes, aspect_ratios, scales):
    default_boxes = []
    for k, (feat_h, feat_w) in enumerate(feat_sizes):
        s_prime_k = math.sqrt(scales[k] * scales[k + 1])
        wh_pairs = [[s_prime_k, s_prime_k]]
        for ar in aspect_ratios[k]:
            sq_ar = math.sqrt(ar)
            w = scales[k] * sq_ar
            h = scales[k] / sq_ar
            wh_pairs.append([w, h])

        shifts_x = ((torch.arange(0, feat_w) + 0.5) / feat_w).to(torch.float32)
        shifts_y = ((torch.arange(0, feat_h) + 0.5) / feat_h).to(torch.float32)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y) * len(wh_pairs), dim=-1).reshape(-1, 2)

        wh_pairs = torch.as_tensor(wh_pairs)
        wh_pairs = wh_pairs.repeat((feat_h * feat_w), 1)
        default_box = torch.cat((shifts, wh_pairs), dim=1)
        default_boxes.append(default_box)

    default_boxes = torch.cat(default_boxes, dim=0)
    default_boxes = torch.cat(
        [
            (default_boxes[:, :2] - 0.5 * default_boxes[:, 2:]),
            (default_boxes[:, :2] + 0.5 * default_boxes[:, 2:]),
        ],
        dim=-1,
    )
    return default_boxes


class SSD(nn.Module):
    def __init__(self, config, num_classes=21):
        super().__init__()
        self.aspect_ratios = config['aspect_ratios']

        self.scales = config['scales']
        self.scales.append(1.0)

        self.num_classes = num_classes
        self.iou_threshold = config['iou_threshold']
        self.low_score_threshold = config['low_score_threshold']
        self.neg_pos_ratio = config['neg_pos_ratio']
        self.pre_nms_topK = config['pre_nms_topK']
        self.nms_threshold = config['nms_threshold']
        self.detections_per_img = config['detections_per_img']

        self.feat_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        default_boxes = generate_default_boxes(self.feat_sizes, self.aspect_ratios, self.scales)
        self.register_buffer('default_boxes', default_boxes)

        self.feat1_enhance = nn.Conv2d(128, 512, kernel_size=1)
        self.feat2_adapter = nn.Conv2d(512, 128, kernel_size=1)
        self.feat2_enhance = nn.Conv2d(256, 1024, kernel_size=1)
        self.feat3_adapter = nn.Conv2d(1024, 256, kernel_size=1)

        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.backbone = backbone

        self.conv_extra1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv_extra2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv_extra3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        out_channels = [512, 1024, 512, 512, 256, 256]

        self.cls_heads = nn.ModuleList()
        for channels, aspect_ratio in zip(out_channels, self.aspect_ratios):
            self.cls_heads.append(
                nn.Conv2d(
                    channels,
                    self.num_classes * (len(aspect_ratio) + 1),
                    kernel_size=3,
                    padding=1
                )
            )

        self.bbox_reg_heads = nn.ModuleList()
        for channels, aspect_ratio in zip(out_channels, self.aspect_ratios):
            self.bbox_reg_heads.append(
                nn.Conv2d(
                    channels,
                    4 * (len(aspect_ratio) + 1),
                    kernel_size=3,
                    padding=1
                )
            )

        for conv_module in [self.conv_extra1, self.conv_extra2, self.conv_extra3]:
            for layer in conv_module.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0.0)

        for module in self.cls_heads:
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
        for module in self.bbox_reg_heads:
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def compute_loss(
            self,
            targets,
            cls_logits,
            bbox_regression,
            default_boxes,
            matched_idxs,
    ):
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for (
            targets_per_image,
            bbox_regression_per_image,
            cls_logits_per_image,
            default_boxes_per_image,
            matched_idxs_per_image,
        ) in zip(targets, bbox_regression, cls_logits, default_boxes, matched_idxs):
            fg_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[
                fg_idxs_per_image
            ]
            num_foreground += foreground_matched_idxs_per_image.numel()

            matched_gt_boxes_per_image = targets_per_image["boxes"][
                foreground_matched_idxs_per_image
            ]
            bbox_regression_per_image = bbox_regression_per_image[fg_idxs_per_image, :]
            default_boxes_per_image = default_boxes_per_image[fg_idxs_per_image, :]
            target_regression = boxes_to_transformation_targets(
                matched_gt_boxes_per_image,
                default_boxes_per_image)

            bbox_loss.append(
                torch.nn.functional.smooth_l1_loss(bbox_regression_per_image,
                                                   target_regression,
                                                   reduction='sum')
            )

            gt_classes_target = torch.zeros(
                (cls_logits_per_image.size(0),),
                dtype=targets_per_image["labels"].dtype,
                device=targets_per_image["labels"].device,
            )
            gt_classes_target[fg_idxs_per_image] = targets_per_image["labels"][
                foreground_matched_idxs_per_image
            ]
            cls_targets.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        num_classes = cls_logits.size(-1)
        cls_loss = torch.nn.functional.cross_entropy(cls_logits.view(-1, num_classes),
                                                     cls_targets.view(-1),
                                                     reduction="none").view(
            cls_targets.size()
        )

        foreground_idxs = cls_targets > 0
        num_negative = self.neg_pos_ratio * foreground_idxs.sum(1, keepdim=True)

        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float("inf")
        values, idx = negative_loss.sort(1, descending=True)
        background_idxs = idx.sort(1)[1] < num_negative
        N = max(1, num_foreground)
        return {
            "bbox_regression": bbox_loss.sum() / N,
            "classification": (cls_loss[foreground_idxs].sum() +
                               cls_loss[background_idxs].sum()) / N,
        }

    def forward(self, x, targets=None):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        feat1 = self.backbone.layer2(x)
        feat1 = self.feat1_enhance(feat1)
        feat2 = self.feat2_adapter(feat1)
        feat2 = self.backbone.layer3(feat2)
        feat2 = self.feat2_enhance(feat2)
        feat3 = self.feat3_adapter(feat2)
        feat3 = self.backbone.layer4(feat3)

        feat4 = self.conv_extra1(feat3)
        feat5 = self.conv_extra2(feat4)
        feat6 = self.conv_extra3(feat5)

        outputs = [feat1, feat2, feat3, feat4, feat5, feat6]

        cls_logits = []
        bbox_reg_deltas = []
        for i, features in enumerate(outputs):
            cls_feat_i = self.cls_heads[i](features)
            bbox_reg_feat_i = self.bbox_reg_heads[i](features)

            N, _, H, W = cls_feat_i.shape
            cls_feat_i = cls_feat_i.view(N, -1, self.num_classes, H, W)
            cls_feat_i = cls_feat_i.permute(0, 3, 4, 1, 2)
            cls_feat_i = cls_feat_i.reshape(N, -1, self.num_classes)
            cls_logits.append(cls_feat_i)

            N, _, H, W = bbox_reg_feat_i.shape
            bbox_reg_feat_i = bbox_reg_feat_i.view(N, -1, 4, H, W)
            bbox_reg_feat_i = bbox_reg_feat_i.permute(0, 3, 4, 1, 2)
            bbox_reg_feat_i = bbox_reg_feat_i.reshape(N, -1, 4)
            bbox_reg_deltas.append(bbox_reg_feat_i)

        cls_logits = torch.cat(cls_logits, dim=1)
        bbox_reg_deltas = torch.cat(bbox_reg_deltas, dim=1)

        default_boxes = [self.default_boxes] * x.size(0)

        losses = {}
        detections = []
        if self.training:
            matched_idxs = []
            for default_boxes_per_image, targets_per_image in zip(default_boxes,
                                                                  targets):
                if targets_per_image["boxes"].numel() == 0:
                    matched_idxs.append(
                        torch.full(
                            (default_boxes_per_image.size(0),), -1,
                            dtype=torch.int64,
                            device=default_boxes_per_image.device
                        )
                    )
                    continue
                iou_matrix = get_iou(targets_per_image["boxes"],
                                     default_boxes_per_image)
                matched_vals, matches = iou_matrix.max(dim=0)

                below_low_threshold = matched_vals < self.iou_threshold
                matches[below_low_threshold] = -1

                _, highest_quality_pred_foreach_gt = iou_matrix.max(dim=1)
                matches[highest_quality_pred_foreach_gt] = torch.arange(
                    highest_quality_pred_foreach_gt.size(0), dtype=torch.int64,
                    device=highest_quality_pred_foreach_gt.device
                )
                matched_idxs.append(matches)
            losses = self.compute_loss(targets, cls_logits, bbox_reg_deltas,
                                      default_boxes, matched_idxs)
        else:
            cls_scores = torch.nn.functional.softmax(cls_logits, dim=-1)
            num_classes = cls_scores.size(-1)

            for bbox_deltas_i, cls_scores_i, default_boxes_i in zip(bbox_reg_deltas,
                                                                    cls_scores,
                                                                    default_boxes):
                boxes = apply_regression_pred_to_default_boxes(bbox_deltas_i,
                                                               default_boxes_i)
                boxes.clamp_(min=0., max=1.)

                pred_boxes = []
                pred_scores = []
                pred_labels = []
                for label in range(1, num_classes):
                    score = cls_scores_i[:, label]

                    keep_idxs = score > self.low_score_threshold
                    score = score[keep_idxs]
                    box = boxes[keep_idxs]

                    score, top_k_idxs = score.topk(min(self.pre_nms_topK, len(score)))
                    box = box[top_k_idxs]

                    pred_boxes.append(box)
                    pred_scores.append(score)
                    pred_labels.append(torch.full_like(score, fill_value=label,
                                                       dtype=torch.int64,
                                                       device=cls_scores.device))

                pred_boxes = torch.cat(pred_boxes, dim=0)
                pred_scores = torch.cat(pred_scores, dim=0)
                pred_labels = torch.cat(pred_labels, dim=0)

                keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
                for class_id in torch.unique(pred_labels):
                    curr_indices = torch.where(pred_labels == class_id)[0]
                    curr_keep_idxs = torch.ops.torchvision.nms(pred_boxes[curr_indices],
                                                               pred_scores[curr_indices],
                                                               self.nms_threshold)
                    keep_mask[curr_indices[curr_keep_idxs]] = True
                keep_indices = torch.where(keep_mask)[0]
                post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(
                    descending=True)[1]]
                keep = post_nms_keep_indices[:self.detections_per_img]
                pred_boxes, pred_scores, pred_labels = (pred_boxes[keep],
                                                        pred_scores[keep],
                                                        pred_labels[keep])

                detections.append(
                    {
                        "boxes": pred_boxes,
                        "scores": pred_scores,
                        "labels": pred_labels,
                    }
                )
        return losses, detections
