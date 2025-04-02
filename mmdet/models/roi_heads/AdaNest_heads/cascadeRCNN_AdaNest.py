
import numpy as np
import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox2result

from mmdet.models.builder import DETECTORS, build_head, build_roi_extractor

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, multi_apply)


from torch import nn, Tensor

from mmcv.cnn.bricks.transformer import  build_positional_encoding

from mmdet.core.bbox.iou_calculators import bbox_overlaps

from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.models.builder import ROI_EXTRACTORS
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import SingleRoIExtractor
from mmdet.models.builder import HEADS

from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead
from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead
from mmdet.core import (bbox2roi,merge_aug_masks)
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import Transformer

@DETECTORS.register_module()
class CascadeRCNN_Ada(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CascadeRCNN_Ada, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def show_result(self, data, result, **kwargs):
        """Show prediction results of the detector.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        return super(CascadeRCNN_Ada, self).show_result(data, result, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)

            # rpn_losses.pop('loss_rpn_bbox')
            losses.update(rpn_losses)
        else:
            proposal_list = proposals


        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        # return self.roi_head.simple_test(
        #     x, [proposal_list[-1][0]], img_metas, rescale=rescale)
        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)




@HEADS.register_module()
class BBoxHead2(Shared2FCBBoxHead):


    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights



@HEADS.register_module()
class AdaNestRoIHead_Multi_stage(CascadeRoIHead):


    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

               Args:
                   bbox_roi_extractor (dict): Config of box roi extractor.
                   bbox_head (dict): Config of box in box head.
               """
        gama = 1
        self.num_stages = 3  # cascade
        self.croin0 = 3
        self.croin1 = 5
        self.croin2 = 6
        self.croin = 9
        self.p = 9
        self.p0 = 7
        self.croip0 = 5
        self.croip1 = 5

        self.emb_min = 64
        self.emb0 = int(256 * 2)
        self.embc = 256

        self.emb_sub = int(256 * 2 * gama)
        self.out_emb = int(1024 * gama)


        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head0 = nn.ModuleList()
        self.bbox_head  = nn.ModuleList()
        self.trans = nn.ModuleList()
        self.env_act0 = nn.ModuleList()
        self.env_act1 = nn.ModuleList()

        self.Convm = nn.ModuleList()
        self.lineC = nn.ModuleList()
        self.Conv_grid = nn.ModuleList()
        self.line_mid0 = nn.ModuleList()
        self.line_mid1 = nn.ModuleList()
        self.line_sub = nn.ModuleList()
        self.crois_pred = nn.ModuleList()

        self.clsc = nn.ModuleList()
        self.regc = nn.ModuleList()
        self.objc = nn.ModuleList()
        self.line_cls = nn.ModuleList()
        self.line_loc = nn.ModuleList()

        self.lineS4 = nn.ModuleList()
        self.lineS5 = nn.ModuleList()
        self.lineS6 = nn.ModuleList()

        self.lineS01 = nn.ModuleList()
        self.lineS02 = nn.ModuleList()
        self.lineS03 = nn.ModuleList()

        self.norm_cls = nn.ModuleList()
        self.norm_loc = nn.ModuleList()
        self.normf = nn.ModuleList()
        self.norm_mid = nn.ModuleList()
        self.norm_sub = nn.ModuleList()

        # self.normz = nn.ModuleList()

        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]

        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages

        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):



            self.bbox_head.append(build_head(head))
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            # self.bbox_head = build_head(head)
            head['type'] = 'BBoxHead_xreg'
            head['num_classes'] = (self.croin + 1)
            head['roi_feat_size'] = 7
            head['fc_out_channels'] = self.out_emb
            # head['reg_class_agnostic'] = True
            head['reg_class_agnostic'] = False
            bbox_head0 = build_head(head)
            bbox_head0.fc_reg = nn.Linear(in_features=self.out_emb, out_features=6 * (self.croin0), bias=True)
            self.bbox_head0.append(bbox_head0)

            # faster_rcnn屏蔽下模块
            # bbox_head_ = build_head(head)
            # self.bbox_head.append(bbox_head_)

            self.crois_pred.append(nn.Linear(self.out_emb, 6 * (self.croin)))
            self.env_act0.append(nn.Linear((256) * 7 * 7, int(1)))
            self.env_act1.append(nn.Linear(self.emb_min * 9 * 9, 1))

            self.Convm.append(nn.Conv2d(256, self.emb_min, 1, 1))
            self.Conv_grid.append(nn.Conv2d(256, self.embc, 1, 1))
            self.lineC.append(nn.Linear(self.emb_min * 9 * 9, self.emb_min * 9 * 9))

            self.line_sub.append(
                nn.Conv1d(self.embc * self.croip1 * self.croip1 * self.croin, int(self.croin * self.emb_sub), 1,
                          groups=self.croin))
            self.line_mid0.append(
                nn.Conv1d(256 * self.croip0 * self.croip0 * self.croin0, int(self.croin0 * (self.emb0)), 1,
                          groups=self.croin0))
            self.line_mid1.append(nn.Linear((self.emb0) * (self.croin0), self.out_emb))
            self.line_cls.append(nn.Linear((self.emb_sub) * (self.croin1), self.out_emb))
            self.line_loc.append(nn.Linear(self.emb_sub * (self.croin2), self.out_emb))

            self.lineS4.append(nn.Linear(self.out_emb, self.out_emb))
            self.lineS5.append(nn.Linear(self.out_emb, self.out_emb))
            self.lineS6.append(nn.Linear(self.out_emb, self.out_emb))

            self.lineS01.append(IdentityLinear(self.out_emb))
            self.lineS02.append(IdentityLinear(self.out_emb))
            self.lineS03.append(IdentityLinear(self.out_emb))

            self.norm_cls.append(nn.BatchNorm1d(self.out_emb))
            self.norm_loc.append(nn.BatchNorm1d(self.out_emb))
            self.normf.append(nn.BatchNorm2d(self.embc))
            self.norm_mid.append(nn.BatchNorm1d((self.emb0) * (self.croin0)))
            self.norm_sub.append(nn.BatchNorm1d((self.emb_sub) * (self.croin)))

            self.clsc.append(nn.Linear(self.out_emb, self.bbox_head[-1].num_classes + 1))
            self.regc.append(nn.Linear(self.out_emb, 4))
            self.objc.append(nn.Linear(self.out_emb, 1))

            ffn_cfgs1 = dict(
                type='FFN',
                embed_dims=256,
                # feedforward_channels=1024,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            )
            encoder = dict(
                type='DetrTransformerEncoder',
                num_layers=2,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)],
                    # feedforward_channels=1024,
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'),
                    ffn_cfgs=ffn_cfgs1
                )
            )

            ffn_cfgs = dict(
                type='FFN',
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            )
            decoder = dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=1,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                            # kdim=256,
                            # vdim=256
                        )],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'),
                    ffn_cfgs=ffn_cfgs),
            )

            # operation_order = ('cross_attn', 'norm', 'ffn', 'norm')
            trans = nn.Sequential()
            for t in range(1):
                trans.append(Transformer_q(encoder=(encoder), decoder=(decoder)))
            self.trans.append(trans)

        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.query_embedding = nn.Embedding(1, self.out_emb)


    def _bbox_forward(self, stage, x, rois,img_shapes):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use

        # stage = 0  # faster_rcnn
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head0 = self.bbox_head0[stage]
        Convm = self.Convm[stage]
        lineC = self.lineC[stage]
        env_act0 = self.env_act0[stage]
        env_act1 = self.env_act1[stage]
        crois_pred = self.crois_pred[stage]

        line_sub = self.line_sub[stage]
        line_mid0 = self.line_mid0[stage]
        line_mid1 = self.line_mid1[stage]
        Conv_grid = self.Conv_grid[stage]

        Line_cls = self.line_cls[stage]
        Line_loc = self.line_loc[stage]
        lineS4 = self.lineS4[stage]
        lineS5 = self.lineS5[stage]
        lineS6 = self.lineS6[stage]
        lineS01 = self.lineS01[stage]
        lineS02 = self.lineS02[stage]
        lineS03 = self.lineS03[stage]
        norm_loc = self.norm_loc[stage]
        norm_cls = self.norm_cls[stage]
        norm_mid = self.norm_mid[stage]
        norm_sub = self.norm_sub[stage]
        normf = self.normf[stage]
        clsc = self.clsc[stage]
        regc = self.regc[stage]
        objc = self.objc[stage]
        trans = self.trans[stage]
        rois_ = rois
        scale_s = 1.2

        rois_scale = self.rois_scale(rois_, scale_s)

        # ===============Environmental_Semantic_Capture=====================
        top_rois_ = torch.arange(0, len(img_shapes)).unsqueeze(1).to(rois)
        top_rois = torch.zeros(len(img_shapes), 4).to(rois)
        for i in range(len(img_shapes)):
            W = img_shapes[i][1]
            H = img_shapes[i][0]
            top_rois[i, 2] = W
            top_rois[i, 3] = H
        top_rois = torch.cat([top_rois_, top_rois], -1)
        top_feats_grid = bbox_roi_extractor(
            x[:bbox_roi_extractor.num_inputs], top_rois)

        x_t = top_feats_grid.flatten(-2).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        ze = torch.zeros(top_feats_grid.size(0), top_feats_grid.size(-2), top_feats_grid.size(-1)).to(x_t) > 0.5
        ke = self.positional_encoding(ze).flatten(-2).movedim(-1, 0)

        top_feats_grid = trans[0].encoder(
            query=x_t,
            key=None,
            value=None,
            query_pos=ke, )

        top_feats_grid = top_feats_grid.movedim(0, -1).unflatten(-1, (9, 9))

        rois_m2 = top_rois[..., 1:].unsqueeze(0)
        rois_m1 = rois_scale[..., 1:].unsqueeze(0).clone().detach()
        xyxy_b = self.sq_k(rois_m1, 1, 9, 9)
        xyxy_b2 = self.sq_k(rois_m2, 1, 9, 9).repeat(1, 1, 1, 1)
        xyxy_b2 = [xyxy_b2[:, [i]].repeat(1, (rois_[..., 0] == i).sum(), 1, 1) for i in range(len(img_shapes))]
        xyxy_b2 = torch.cat(xyxy_b2, 1)
        cosover4 = bbox_overlaps(xyxy_b, xyxy_b2, 'iof')
        cosover4 = cosover4.movedim(0, 1).flatten(1, 2)
        x_roi4 = top_feats_grid.flatten(-2)
        x_roi4 = [x_roi4[[i]].repeat((rois_scale[..., 0] == i).sum(), 1, 1) for i in range(len(img_shapes))]
        x_roi4 = torch.cat(x_roi4, 0)

        env_feats_grid = cosover4.bmm(x_roi4.transpose(-1, -2)).unflatten(1, (1, -1)).movedim(1, 0).transpose(
            -1, -2).unflatten(-1, (9, 9)).squeeze(0)

        env_s = env_act0(env_feats_grid[..., 1:8, 1:8].flatten(1))
        # ====================================================================

        if stage <= 2:
            # ================Roi_sampling==================================
            feats_grid = bbox_roi_extractor(
                x[:bbox_roi_extractor.num_inputs], rois_scale)

            if self.with_shared_head:
                feats_grid = self.shared_head(feats_grid)

            feats_gridm0 = Convm(feats_grid) + feats_grid.unflatten(1, (-1, 4)).sum(2)

            env_s = env_s.clone() + env_act1(feats_gridm0.flatten(-3))
            env_feats_grid = env_feats_grid.clone() * env_s.sigmoid().unsqueeze(-1).unsqueeze(-1)
            feats_grid = feats_grid.clone() + env_feats_grid
            # bbox_featsm0 = lineS(bbox_feats) + bbox_feats.unflatten(1, (-1, 4)).sum(2)+bbox_feats_topd
            feats_gridm1 = feats_gridm0 + lineC(feats_gridm0.flatten(-3)).reshape(-1, self.emb_min, 9, 9)

            feats_grid = (feats_grid.unflatten(1, (-1, 4)) + feats_gridm1.unsqueeze(2)).flatten(1, 2)
            feats_grid_ = feats_grid[..., 1:8, 1:8]
            x_1, mid_bbox_pred = bbox_head0(feats_grid_)
            # ================================================================================

            # ================inter_multi_subregion_sampling==================================
            mid_bbox_pred = mid_bbox_pred.unflatten(1, (-1, 6)).movedim(1, 0)
            scale_s = 1

            rois_plus1 = self.rois_scale(rois_, scale_s)
            all_bbox_preds1 = bbox_xyxy_to_cxcywh(rois_plus1[:, 1:])
            all_bbox_preds2 = torch.zeros_like(mid_bbox_pred)
            all_bbox_preds2[..., 0:2] = all_bbox_preds1[..., 0:2] + 1 * all_bbox_preds1[..., 2:] * (
                    (mid_bbox_pred[..., 0:2]).sigmoid() - 0.5)
            pp = 0.4 * torch.ones([self.croin0, 1, 1]).to(mid_bbox_pred)
            # pp[self.croin1:self.croin, :, :] = 0.35

            all_bbox_preds2[..., 2:4] = (all_bbox_preds1[..., 2:4] * pp * (
                    (mid_bbox_pred[..., 2:4].clamp(-6.60, 4)).exp() + 0.01)) + 1
            all_bbox_preds2[..., 4:] = (all_bbox_preds1[..., 2:4] * pp * (
                    (mid_bbox_pred[..., 4:].clamp(-6.60, 4)).exp() + 0.01)) + 1
            rois_scale_rep = rois_scale.unsqueeze(0).repeat(self.croin0, 1, 1)
            all_bbox_preds3c = torch.zeros_like(rois_scale_rep[..., 1:])
            all_bbox_preds3c[..., 0] = (all_bbox_preds2[..., 0] - all_bbox_preds2[..., 2] / 2).clamp(
                rois_scale_rep[..., 1], rois_scale_rep[..., 3])
            all_bbox_preds3c[..., 1] = (all_bbox_preds2[..., 1] - all_bbox_preds2[..., 3] / 2).clamp(
                rois_scale_rep[..., 2], rois_scale_rep[..., 4])
            all_bbox_preds3c[..., 2] = (all_bbox_preds2[..., 0] + all_bbox_preds2[..., 4] / 2).clamp(
                rois_scale_rep[..., 1], rois_scale_rep[..., 3])
            all_bbox_preds3c[..., 3] = (all_bbox_preds2[..., 1] + all_bbox_preds2[..., 5] / 2).clamp(
                rois_scale_rep[..., 2], rois_scale_rep[..., 4])

            mid_rois = torch.cat((rois_scale_rep[..., :1], all_bbox_preds3c), dim=-1)
            for i in range(len(img_shapes)):
                W = img_shapes[i][1]
                H = img_shapes[i][0]
                ind = rois[..., 0] == i
                mid_rois[:, ind, 1] = mid_rois[:, ind, 1].clamp(0, W)
                mid_rois[:, ind, 2] = mid_rois[:, ind, 2].clamp(0, H)
                mid_rois[:, ind, 3] = mid_rois[:, ind, 3].clamp(0, W)
                mid_rois[:, ind, 4] = mid_rois[:, ind, 4].clamp(0, H)

            mid_rois[..., 1:] = bbox_xyxy_to_cxcywh(mid_rois[..., 1:])
            mid_rois[..., 3:] = mid_rois[..., 3:].relu() + 1
            mid_rois[..., 1:] = bbox_cxcywh_to_xyxy(mid_rois[..., 1:])

            xyxy_b = self.sq_k(mid_rois[:self.croin0, ..., 1:], 1, self.croip0, self.croip0)
            xyxy_b2 = self.sq_k(rois_scale[..., 1:].unsqueeze(0).clone().detach(), 1, 9, 9).repeat(self.croin0, 1, 1, 1)
            cosover1 = bbox_overlaps(xyxy_b, xyxy_b2, 'iof')
            cosover1 = cosover1.movedim(0, 1).flatten(1, 2)
            x_roi1 = feats_grid.flatten(-2)
            mid_feats_grid = cosover1.bmm(x_roi1.transpose(-1, -2)).unflatten(1, (self.croin0, -1)).movedim(1,
                                                                                                            0).transpose(
                -1, -2).flatten(-2).movedim(0, 1).flatten(1).unsqueeze(-1)
            # bbox_featsg = cosover1.bmm(x_roi1.transpose(-1, -2)).unflatten(1,(self.croin0,-1)).movedim(1,0).transpose(-1, -2).unflatten(-1, (self.croip0, self.croip0))
            mid_feats = line_mid0(mid_feats_grid).squeeze(-1)
            mid_feats = norm_mid(mid_feats)

            mid_f = line_mid1(mid_feats)
            mid_f_reg = (x_1 + lineS6(x_1) + mid_f).relu()
            # ===========================================================================

            # ================multi-subregion_generation==================================
            sub_bbox_pred = crois_pred(mid_f_reg)
            sub_bbox_pred = sub_bbox_pred.unflatten(1, (-1, 6)).movedim(1, 0)
            scale_s = 1
            rois_plus1 = self.rois_scale(rois_, scale_s)
            all_bbox_preds1 = bbox_xyxy_to_cxcywh(rois_plus1[:, 1:])
            all_bbox_preds2 = torch.zeros_like(sub_bbox_pred)
            all_bbox_preds2[..., 0:2] = all_bbox_preds1[..., 0:2] + 1 * all_bbox_preds1[..., 2:] * (
                    (sub_bbox_pred[..., 0:2]).sigmoid() - 0.5)
            pp = torch.zeros([self.croin, 1, 1]).to(sub_bbox_pred)

            pp[:, :, :] = 0.33

            ppd = torch.zeros_like(pp)
            ppd[:, :, :] = -6.6

            ppu = 4 * torch.ones_like(pp)
            ppd = ppd.repeat(1, rois.size(0), 2)
            ppu = ppu.repeat(1, rois.size(0), 2)

            all_bbox_preds2[..., 2:4] = (all_bbox_preds1[..., 2:4] * pp * (
                    (sub_bbox_pred[..., 2:4].clamp(ppd, ppu)).exp() + 0.01)) + 1
            all_bbox_preds2[..., 4:] = (all_bbox_preds1[..., 2:4] * pp * (
                    (sub_bbox_pred[..., 4:].clamp(ppd, ppu)).exp() + 0.01)) + 1

            rois_scale_rep = rois_scale.unsqueeze(0).repeat(self.croin, 1, 1)
            all_bbox_preds3c = torch.zeros_like(rois_scale_rep[..., 1:])
            all_bbox_preds3c[..., 0] = (all_bbox_preds2[..., 0] - all_bbox_preds2[..., 2] / 2).clamp(
                rois_scale_rep[..., 1], rois_scale_rep[..., 3])
            all_bbox_preds3c[..., 1] = (all_bbox_preds2[..., 1] - all_bbox_preds2[..., 3] / 2).clamp(
                rois_scale_rep[..., 2], rois_scale_rep[..., 4])
            all_bbox_preds3c[..., 2] = (all_bbox_preds2[..., 0] + all_bbox_preds2[..., 4] / 2).clamp(
                rois_scale_rep[..., 1], rois_scale_rep[..., 3])
            all_bbox_preds3c[..., 3] = (all_bbox_preds2[..., 1] + all_bbox_preds2[..., 5] / 2).clamp(
                rois_scale_rep[..., 2], rois_scale_rep[..., 4])

            sub_rois = torch.cat((rois_scale_rep[..., :1], all_bbox_preds3c), dim=-1)

            for i in range(len(img_shapes)):
                W = img_shapes[i][1]
                H = img_shapes[i][0]
                ind = rois[..., 0] == i
                sub_rois[:, ind, 1] = sub_rois[:, ind, 1].clamp(0, W)
                sub_rois[:, ind, 2] = sub_rois[:, ind, 2].clamp(0, H)
                sub_rois[:, ind, 3] = sub_rois[:, ind, 3].clamp(0, W)
                sub_rois[:, ind, 4] = sub_rois[:, ind, 4].clamp(0, H)

            sub_rois[..., 1:] = bbox_xyxy_to_cxcywh(sub_rois[..., 1:])
            sub_rois[..., 3:] = sub_rois[..., 3:].relu() + 1
            sub_rois[..., 1:] = bbox_cxcywh_to_xyxy(sub_rois[..., 1:])
            # ================multi-subregion_generation==================================

            feats_gridm = normf(Conv_grid(feats_grid))
            # ================multi-subregion_sampling==================================
            xyxy_b = self.sq_k(sub_rois[..., 1:], 1, self.croip1, self.croip1)
            xyxy_b2 = self.sq_k(rois_scale[..., 1:].unsqueeze(0).clone().detach(), 1, 9, 9).repeat(self.croin, 1, 1, 1)
            cosover2 = bbox_overlaps(xyxy_b, xyxy_b2, 'iof')
            cosover2 = cosover2.movedim(0, 1).flatten(1, 2)
            x_roi2 = feats_gridm.flatten(-2)
            sub_feats_grid = cosover2.bmm(x_roi2.transpose(-1, -2)).unflatten(1, (self.croin, -1)).movedim(1,
                                                                                                           0).transpose(
                -1, -2).unflatten(-1, (self.croip1, self.croip1)).flatten(2).movedim(0, 1).flatten(1).unsqueeze(-1)
            sub_feats = line_sub(sub_feats_grid).squeeze(-1)
            sub_feats = norm_sub(sub_feats).relu()
            # ===========================================================================

        # =========================Semantic Separation Strategy and Decoupled Prediction===============
        b0 = Line_cls(sub_feats[..., :self.croin1 * self.emb_sub])
        b0 = norm_cls(b0)
        b1 = Line_loc(sub_feats[..., -self.croin2 * self.emb_sub:])
        b1 = norm_loc(b1)
        x_1 = x_1 + lineS03(mid_f)
        xw0 = (b0 + lineS4(b0) + x_1 + lineS01(x_1)).relu()
        xw1 = (b1 + lineS5(b1) + x_1 + lineS02(x_1)).relu()
        cls_score = clsc(xw0)
        xs = objc(xw1)
        bbox_predc = regc(xw1)
        cls_score[..., :-1] = cls_score.softmax(-1)[..., :-1] * xs[..., [-1]].sigmoid() + 0.0
        cls_score[..., -1] = 1 - cls_score[..., :-1].sum(-1)
        cls_score += 0.0001
        cls_score = cls_score.log()

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_predc, bbox_feats=sub_feats, object_s=xs.sigmoid())
        return bbox_results
    def rois_scale(self, rois_, scale_s):
        all_bbox_preds1 = torch.zeros_like(rois_)
        all_bbox_preds1[:, 1] = (rois_[:, 1] + rois_[:, 3]) / 2
        all_bbox_preds1[:, 2] = (rois_[:, 2] + rois_[:, 4]) / 2
        all_bbox_preds1[:, 3] = (rois_[:, 3] - rois_[:, 1]).relu() + 1
        all_bbox_preds1[:, 4] = (rois_[:, 4] - rois_[:, 2]).relu() + 1
        all_bbox_preds3 = torch.zeros_like(rois_)
        all_bbox_preds3[..., 0] = rois_[..., 0]
        all_bbox_preds3[..., 1] = all_bbox_preds1[..., 1] - scale_s * all_bbox_preds1[..., 3] / 2
        all_bbox_preds3[..., 2] = all_bbox_preds1[..., 2] - scale_s * all_bbox_preds1[..., 4] / 2
        all_bbox_preds3[..., 3] = all_bbox_preds1[..., 1] + scale_s * all_bbox_preds1[..., 3] / 2
        all_bbox_preds3[..., 4] = all_bbox_preds1[..., 2] + scale_s * all_bbox_preds1[..., 4] / 2
        rois2_ = all_bbox_preds3.clone().detach()
        return rois2_
    def sq_k(self, bbox_, bei, ku,ku1):
        x_l = torch.arange(0, ku).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(bbox_.size(0),bbox_.size(1), ku1, 1).to(bbox_)
        x_r = x_l + 1
        y_u = torch.arange(0, ku1).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bbox_.size(0),bbox_.size(1), 1, ku).to(bbox_)
        y_d = y_u + 1
        jian = (bbox_[..., 2] - bbox_[..., 0]) / ku
        jian2 = (bbox_[..., 3] - bbox_[..., 1]) / ku1
        x_l = x_l * jian.unsqueeze(-1).unsqueeze(-1) + bbox_[..., 0].unsqueeze(-1).unsqueeze(-1)
        x_r = x_r * jian.unsqueeze(-1).unsqueeze(-1) + bbox_[..., 0].unsqueeze(-1).unsqueeze(-1)
        y_u = y_u * jian2.unsqueeze(-1).unsqueeze(-1) + bbox_[..., 1].unsqueeze(-1).unsqueeze(-1)
        y_d = y_d * jian2.unsqueeze(-1).unsqueeze(-1) + bbox_[..., 1].unsqueeze(-1).unsqueeze(-1)
        xyxy_b = torch.stack([x_l.flatten(-2), y_u.flatten(-2), x_r.flatten(-2), y_d.flatten(-2)], -1)
        return xyxy_b
    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois,img_shapes)
            # bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [
                        self.bbox_head[i].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refined_rois = self.bbox_head[i].regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    mask_pred = mask_results['mask_pred']
                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append([
                        m.sigmoid().cpu().detach().numpy() for m in mask_pred
                    ])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    rcnn_train_cfg,img_metas)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(
                    i, x, sampling_results, gt_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'])
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(
                            cls_score)

                    # Empty proposal.
                    if cls_score.numel() == 0:
                        break

                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[i].num_classes,
                        cls_score[:, :-1].argmax(1), roi_labels)
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

        return losses

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg,img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)

        bbox_results = self._bbox_forward(stage, x, rois,img_shapes)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results



class IdentityLinear(nn.Module):
    def __init__(self, in_features):
        super(IdentityLinear, self).__init__()
        self.linear = nn.Linear(in_features, in_features)

        # 将权重初始化为单位矩阵
        # nn.init.eye_(self.linear.weight)
        nn.init.constant_(self.linear.weight, 0)

        # 将偏置初始化为零
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.linear(x)

    # def backward(self, gradient_output):
    #     # 阻止梯度的更新，将其清零
    #     if self.linear.weight.grad is not None:
    #         self.linear.weight.grad.zero_()
    #     if self.linear.bias is not None and self.linear.bias.grad is not None:
    #         self.linear.bias.grad.zero_()



@HEADS.register_module()
class BBoxHead_xreg(Shared2FCBBoxHead):

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return x, bbox_pred


@TRANSFORMER.register_module()
class Transformer_q(Transformer):
    """Implements the DETR transformer.

    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:

        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """



    def forward(self, x, mask, query_embed, pos_embed):
        """Forward function for `Transformer`.

        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, c, h, w = x.shape
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        x = x.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        # query_embed = query_embed.unsqueeze(1).repeat(
        #     1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        # out_dec = self.decoder(
        #     query=target,
        #     key=memory,
        #     value=memory,
        #     key_pos=pos_embed,
        #     query_pos=query_embed,
        #     key_padding_mask=mask)
        out_dec = self.decoder(
            query=query_embed,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=target,
            key_padding_mask=mask)
        out_dec = out_dec.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        return out_dec, memory