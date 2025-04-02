import torch
from mmdet.models.builder import build_head, build_roi_extractor
from torch import nn
from mmcv.cnn.bricks.transformer import  build_positional_encoding
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.core import bbox2roi
from mmdet.models.utils.transformer import Transformer
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh,bbox_cxcywh_to_xyxy
from mmdet.models.losses import accuracy

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
@HEADS.register_module()
class BBoxHeadsimple(Shared2FCBBoxHead):
    def forward(self, x):

        x_cls = x[:,:int(x.size(-1)/2)]
        x_reg = x[:,int(x.size(-1)/2):]

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

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module()
class AdaNestRoIHead_1stage(StandardRoIHead):

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

               Args:
                   bbox_roi_extractor (dict): Config of box roi extractor.
                   bbox_head (dict): Config of box in box head.
               """
        gama = 1
        self.croin0 = 3
        self.croin1 = 5
        self.croin2 = 6
        self.croin = 9
        self.p = 9
        self.p0 = 7
        self.croip0 = 5
        self.croip1 = 5

        self.emb_min=64
        self.emb0 = int(256 * 2)
        self.embc =256

        self.emb_sub = int(256*2*gama)
        self.out_emb = int(1024*gama)


        self.num_stages = 1 #这是faster只有一次调整
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head0 = nn.ModuleList()

        self.trans = nn.ModuleList()
        self.env_act0 = nn.ModuleList()
        self.env_act1= nn.ModuleList()

        self.Convm = nn.ModuleList()
        self.lineC= nn.ModuleList()
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
        self.numj = 0
        # self.normz = nn.ModuleList()

        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]

        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages

        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):

            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head= build_head(head)
            head['type'] = 'BBoxHead_xreg'
            head['num_classes'] = (self.croin+1)
            head['roi_feat_size'] = 7
            head['fc_out_channels'] = self.out_emb
            # head['reg_class_agnostic'] = True
            head['reg_class_agnostic'] = False
            bbox_head0 = build_head(head)
            bbox_head0.fc_reg = nn.Linear(in_features=self.out_emb, out_features=6*(self.croin0), bias=True)
            self.bbox_head0.append(bbox_head0)



            #faster_rcnn屏蔽下模块
            # bbox_head_ = build_head(head)
            # self.bbox_head.append(bbox_head_)

            self.crois_pred.append(nn.Linear(self.out_emb, 6 * (self.croin)))
            self.env_act0.append(nn.Linear((256) * 7 * 7, int(1)))
            self.env_act1.append(nn.Linear(self.emb_min * 9 * 9, 1))

            self.Convm.append(nn.Conv2d(256,self.emb_min,1,1))
            self.Conv_grid.append(nn.Conv2d(256,self.embc,1,1))
            self.lineC.append(nn.Linear(self.emb_min * 9 * 9, self.emb_min * 9 * 9))


            self.line_sub.append(nn.Conv1d(self.embc * self.croip1 * self.croip1 * self.croin, int(self.croin * self.emb_sub), 1, groups=self.croin))
            self.line_mid0.append(nn.Conv1d(256 * self.croip0 * self.croip0 * self.croin0, int(self.croin0 * (self.emb0)), 1, groups=self.croin0))
            self.line_mid1.append(nn.Linear((self.emb0)*(self.croin0), self.out_emb))
            self.line_cls.append(nn.Linear((self.emb_sub)*(self.croin1), self.out_emb))
            self.line_loc.append(nn.Linear(self.emb_sub*(self.croin2), self.out_emb))


            self.lineS4.append(nn.Linear(self.out_emb, self.out_emb))
            self.lineS5.append(nn.Linear(self.out_emb, self.out_emb))
            self.lineS6.append(nn.Linear(self.out_emb, self.out_emb))

            self.lineS01.append(IdentityLinear(self.out_emb))
            self.lineS02.append(IdentityLinear(self.out_emb))
            self.lineS03.append(IdentityLinear(self.out_emb))

            self.norm_cls.append(nn.BatchNorm1d(self.out_emb))
            self.norm_loc.append(nn.BatchNorm1d(self.out_emb))
            self.normf.append(nn.BatchNorm2d(self.embc))
            self.norm_mid.append(nn.BatchNorm1d((self.emb0)*(self.croin0)))
            self.norm_sub.append(nn.BatchNorm1d((self.emb_sub)*(self.croin)))

            self.clsc.append(nn.Linear(self.out_emb, self.bbox_head.num_classes+1))
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

    def _bbox_forward(self, x, rois,img_shapes):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use

        stage = 0 #faster_rcnn
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head0 = self.bbox_head0[stage]
        Convm = self.Convm[stage]
        lineC= self.lineC[stage]
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
        norm_loc= self.norm_loc[stage]
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

        #===============Environmental_Semantic_Capture=====================
        top_rois_ = torch.arange(0, len(img_shapes)).unsqueeze(1).to(rois)
        top_rois = torch.zeros(len(img_shapes),4).to(rois)
        for i in range(len(img_shapes)):
            W = img_shapes[i][1]
            H = img_shapes[i][0]
            top_rois[i,2] = W
            top_rois[i,3] = H
        top_rois = torch.cat([top_rois_,top_rois],-1)
        top_feats_grid = bbox_roi_extractor(
            x[:bbox_roi_extractor.num_inputs], top_rois)


        x_t = top_feats_grid.flatten(-2).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        ze = torch.zeros(top_feats_grid.size(0),top_feats_grid.size(-2),top_feats_grid.size(-1)).to(x_t)>0.5
        ke = self.positional_encoding(ze).flatten(-2).movedim(-1,0)

        top_feats_grid = trans[0].encoder(
            query=x_t,
            key=None,
            value=None,
            query_pos=ke,)

        top_feats_grid = top_feats_grid.movedim(0,-1).unflatten(-1,(9,9))

        rois_m2 = top_rois[..., 1:].unsqueeze(0)
        rois_m1 = rois_scale[..., 1:].unsqueeze(0).clone().detach()
        xyxy_b = self.sq_k(rois_m1, 1, 9, 9)
        xyxy_b2 = self.sq_k(rois_m2, 1, 9, 9).repeat(1, 1, 1, 1)
        xyxy_b2 = [xyxy_b2[:,[i]].repeat(1,(rois_[..., 0]==i).sum(),1,1) for i in range(len(img_shapes))]
        xyxy_b2 = torch.cat(xyxy_b2,1)
        cosover4 = bbox_overlaps(xyxy_b, xyxy_b2, 'iof')
        cosover4 = cosover4.movedim(0, 1).flatten(1, 2)
        x_roi4 = top_feats_grid.flatten(-2)
        x_roi4 = [x_roi4[[i]].repeat((rois_scale[..., 0]==i).sum(),1,1) for i in range(len(img_shapes))]
        x_roi4 = torch.cat(x_roi4, 0)
        
        env_feats_grid = cosover4.bmm(x_roi4.transpose(-1, -2)).unflatten(1, (1, -1)).movedim(1, 0).transpose(
            -1, -2).unflatten(-1, (9, 9)).squeeze(0)

        env_s = env_act0(env_feats_grid[...,1:8,1:8].flatten(1))
        #====================================================================

        if stage <= 2:
            # ================Roi_sampling==================================
            feats_grid = bbox_roi_extractor(
                x[:bbox_roi_extractor.num_inputs], rois_scale)

            if self.with_shared_head:
                feats_grid = self.shared_head(feats_grid)

            feats_gridm0 = Convm(feats_grid) + feats_grid.unflatten(1, (-1, 4)).sum(2)

            env_s =env_s.clone() + env_act1(feats_gridm0.flatten(-3))
            env_feats_grid = env_feats_grid.clone() * env_s.sigmoid().unsqueeze(-1).unsqueeze(-1)
            feats_grid =feats_grid.clone() +  env_feats_grid
            # bbox_featsm0 = lineS(bbox_feats) + bbox_feats.unflatten(1, (-1, 4)).sum(2)+bbox_feats_topd
            feats_gridm1 = feats_gridm0 + lineC(feats_gridm0.flatten(-3)).reshape(-1,self.emb_min,9,9)

            feats_grid = (feats_grid.unflatten(1,(-1,4))+feats_gridm1.unsqueeze(2)).flatten(1,2)
            feats_grid_ = feats_grid[...,1:8,1:8]
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
            all_bbox_preds1 = bbox_xyxy_to_cxcywh(rois_plus1[:,1:])
            all_bbox_preds2 = torch.zeros_like(sub_bbox_pred)
            all_bbox_preds2[..., 0:2] = all_bbox_preds1[..., 0:2] + 1 * all_bbox_preds1[..., 2:] * (
                    (sub_bbox_pred[..., 0:2]).sigmoid() - 0.5)
            pp = torch.zeros([self.croin,1,1]).to(sub_bbox_pred)

            pp[:,:,:] = 0.33

            ppd = torch.zeros_like(pp)
            ppd[:,:,:] = -6.6

            ppu = 4*torch.ones_like(pp)
            ppd = ppd.repeat(1,rois.size(0),2)
            ppu = ppu.repeat(1,rois.size(0),2)

            all_bbox_preds2[..., 2:4] = (all_bbox_preds1[..., 2:4] * pp * (
                    (sub_bbox_pred[..., 2:4].clamp(ppd, ppu)).exp() + 0.01)) + 1
            all_bbox_preds2[..., 4:] = (all_bbox_preds1[..., 2:4] * pp * (
                    (sub_bbox_pred[..., 4:].clamp(ppd, ppu)).exp() + 0.01)) + 1

            rois_scale_rep = rois_scale.unsqueeze(0).repeat(self.croin, 1, 1)
            all_bbox_preds3c = torch.zeros_like(rois_scale_rep[...,1:])
            all_bbox_preds3c[..., 0] = (all_bbox_preds2[..., 0] - all_bbox_preds2[..., 2] / 2).clamp(rois_scale_rep[..., 1], rois_scale_rep[..., 3])
            all_bbox_preds3c[..., 1] = (all_bbox_preds2[..., 1] - all_bbox_preds2[..., 3] / 2).clamp(rois_scale_rep[..., 2], rois_scale_rep[..., 4])
            all_bbox_preds3c[..., 2] = (all_bbox_preds2[..., 0] + all_bbox_preds2[..., 4] / 2).clamp(rois_scale_rep[..., 1], rois_scale_rep[..., 3])
            all_bbox_preds3c[..., 3] = (all_bbox_preds2[..., 1] + all_bbox_preds2[..., 5] / 2).clamp(rois_scale_rep[..., 2], rois_scale_rep[..., 4])

            sub_rois = torch.cat((rois_scale_rep[...,:1], all_bbox_preds3c), dim=-1)


            for i in range(len(img_shapes)):
                W = img_shapes[i][1]
                H = img_shapes[i][0]
                ind = rois[...,0] == i
                sub_rois[:,ind, 1] = sub_rois[:,ind, 1].clamp(0, W)
                sub_rois[:,ind, 2] = sub_rois[:,ind, 2].clamp(0, H)
                sub_rois[:,ind, 3] = sub_rois[:,ind, 3].clamp(0, W)
                sub_rois[:,ind, 4] = sub_rois[:,ind, 4].clamp(0, H)

            sub_rois[...,1:] = bbox_xyxy_to_cxcywh(sub_rois[...,1:])
            sub_rois[...,3:] = sub_rois[...,3:].relu()+1
            sub_rois[..., 1:] = bbox_cxcywh_to_xyxy(sub_rois[..., 1:])
            # ================multi-subregion_generation==================================


            feats_gridm = normf(Conv_grid(feats_grid))
            #================multi-subregion_sampling==================================
            xyxy_b = self.sq_k(sub_rois[..., 1:], 1, self.croip1, self.croip1)
            xyxy_b2 = self.sq_k(rois_scale[..., 1:].unsqueeze(0).clone().detach(), 1, 9, 9).repeat(self.croin, 1, 1,1)
            cosover2 = bbox_overlaps(xyxy_b, xyxy_b2, 'iof')
            cosover2 = cosover2.movedim(0,1).flatten(1,2)
            x_roi2 = feats_gridm.flatten(-2)
            sub_feats_grid = cosover2.bmm(x_roi2.transpose(-1, -2)).unflatten(1, (self.croin, -1)).movedim(1, 0).transpose(
                -1, -2).unflatten(-1, (self.croip1, self.croip1)).flatten(2).movedim(0,1).flatten(1).unsqueeze(-1)
            sub_feats = line_sub(sub_feats_grid).squeeze(-1)
            sub_feats = norm_sub(sub_feats).relu()
            #===========================================================================

        #=========================Semantic Separation Strategy and Decoupled Prediction===============
        b0 = Line_cls(sub_feats[...,:self.croin1*self.emb_sub])
        b0 = norm_cls(b0)
        b1 = Line_loc(sub_feats[...,-self.croin2*self.emb_sub:])
        b1 = norm_loc(b1)
        x_1 = x_1 + lineS03(mid_f)
        xw0 = (b0+lineS4(b0) + x_1+lineS01(x_1)).relu()
        xw1 = (b1+lineS5(b1) + x_1+lineS02(x_1)).relu()
        cls_score = clsc(xw0)
        xs = objc(xw1)
        bbox_predc = regc(xw1)
        cls_score[..., :-1] = cls_score.softmax(-1)[..., :-1] * xs[..., [-1]].sigmoid() + 0.0
        cls_score[..., -1] = 1 - cls_score[..., :-1].sum(-1)
        cls_score += 0.0001
        cls_score = cls_score.log()

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_predc, bbox_feats=feats_grid, object_s=xs.sigmoid(),subregion = sub_rois)
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

    def split_rois(self,rois):
        """
        将每个 RoI (xyxy) 划分为 9 个子边界框，包括：
        1. 4 个田字四等分子边界框
        2. 2 个上下二等分子边界框
        3. 2 个左右二等分子边界框
        4. 1 个中心区域，面积为原边界框的 1/2

        Args:
            rois (Tensor): 形状为 (N, 5) 的张量，其中第 0 列为辅助维度，剩余 4 列表示边界框 (x1, y1, x2, y2)。

        Returns:
            Tensor: 形状为 (9, N, 5) 的张量，每个 RoI 被分割为 9 个子边界框，并保留辅助维度。
        """
        roi_ids = rois[:, 0].unsqueeze(0).expand(9, -1)  # 复制辅助维度到 9 个子边界框
        x1, y1, x2, y2 = rois[:, 1], rois[:, 2], rois[:, 3], rois[:, 4]

        w_half = (x2 - x1) / 2.0
        h_half = (y2 - y1) / 2.0
        w_quarter = (x2 - x1) / 4.0
        h_quarter = (y2 - y1) / 4.0

        # 计算九个子边界框的坐标
        sub_boxes = torch.stack([
            torch.stack([x1, y1, x1 + w_half, y1 + h_half], dim=-1),  # 左上
            torch.stack([x1 + w_half, y1, x2, y1 + h_half], dim=-1),  # 右上
            torch.stack([x1, y1 + h_half, x1 + w_half, y2], dim=-1),  # 左下
            torch.stack([x1 + w_half, y1 + h_half, x2, y2], dim=-1),  # 右下
            torch.stack([x1, y1, x2, y1 + h_half], dim=-1),  # 上半部分
            torch.stack([x1, y1 + h_half, x2, y2], dim=-1),  # 下半部分
            torch.stack([x1, y1, x1 + w_half, y2], dim=-1),  # 左半部分
            torch.stack([x1 + w_half, y1, x2, y2], dim=-1),  # 右半部分
            torch.stack([x1 + w_quarter, y1 + h_quarter, x2 - w_quarter, y2 - h_quarter], dim=-1)  # 中间部分
        ], dim=0)  # 形状 (9, N, 4)

        sub_boxes = torch.cat([roi_ids.unsqueeze(-1), sub_boxes], dim=-1)  # 添加辅助维度，形状 (9, N, 5)

        return sub_boxes

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False):
        """Test only det bboxes without augmentation.

         Args:
             x (tuple[Tensor]): Feature maps of all scale level.
             img_metas (list[dict]): Image meta info.
             proposals (List[Tensor]): Region proposals.
             rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
             rescale (bool): If True, return boxes in original image space.
                 Default: False.

         Returns:
             tuple[list[Tensor], list[Tensor]]: The first list contains
                 the boxes of the corresponding image in a batch, each
                 tensor has the shape (num_boxes, 5) and last dimension
                 5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                 in the second list is the labels with shape (num_boxes, ).
                 The length of both lists should be equal to batch_size.
         """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0,), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        bbox_results = self._bbox_forward(x, rois,img_shapes)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0,), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        bbox_results = self._bbox_forward(x, rois,img_shapes)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)


        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results


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
        mask = mask.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
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


def compute_area_batch(boxes):
    """
    计算批量边界框的面积。

    参数:
    - boxes: Tensor, 形状为 (1, N, 4)，N个边界框，每个边界框由 [x_min, y_min, x_max, y_max] 表示

    返回:
    - area: Tensor, 形状为 (1, N)，每个边界框的面积
    """
    # 计算宽度和高度
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]

    # 计算面积
    area = width * height
    return area