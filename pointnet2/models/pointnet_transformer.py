
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import etw_pytorch_utils as pt_utils
from .transformer import TransformerDecoder, TransformerEncoder
from .multihead_attention import MultiheadAttention
from pointnet2.utils.pointnet2_modules import PointnetSAModule
from pointnet2.utils import pointnet2_utils


class Pointnet_Backbone(nn.Module):
    def __init__(self, input_channels=3,
                 use_xyz=True, sample_method=None,
                 first_sample_method=None):
        super(Pointnet_Backbone, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.3,
                nsample=32,
                mlp=[input_channels, 64, 64, 128],
                use_xyz=use_xyz,
                sample_method=first_sample_method
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.5,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz, # False,
                sample_method=sample_method
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.7,
                nsample=32,
                mlp=[256, 256, 256, 256],
                use_xyz=use_xyz, # False,
                sample_method=sample_method,
            )
        )
        self.cov_final = nn.Conv1d(256, 256, kernel_size=1)
        self.sample_method = sample_method


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, numpoints,
                target_features=None,
                keep_first_half=False):
        # type: (Pointnet2SSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        idxs = []
        for i in range(len(self.SA_modules)):
            if i > 0 and target_features is not None:
                assert not keep_first_half
                li_xyz, li_features, idx = self.SA_modules[i](
                    l_xyz[i], l_features[i], numpoints[i],
                    target_feature=target_features[i])
            else:
                li_xyz, li_features, idx = self.SA_modules[i](
                    l_xyz[i], l_features[i], numpoints[i],
                    keep_first_half=keep_first_half)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            idxs.append(idx)
        output_dict = {
            'xyz' : l_xyz[-1],
            'feature' : self.cov_final(l_features[-1]),
            'idxs' : idxs,
            'xyzs' : l_xyz,
            'features' : l_features
        }
        return output_dict


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel=3, num_pos_feats=256):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        # xyz : BxNx3
        xyz = xyz.transpose(1, 2).contiguous()
        # Bx3xN
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class PointnetTransformerSiamese(nn.Module):
    def __init__(self, input_channels=3,
                 use_xyz=True,
                 objective = False,
                 input_size=1024,
                 sample_method=None):
        super(PointnetTransformerSiamese, self).__init__()
        self.input_size = input_size

        sample_method = None # 'ffps'
        vote_sample_method = None # 'ffps'
        first_sample_method = None
        self.sample_method = sample_method
        d_model = 256
        num_layers = 1
        self.with_pos_embed = True

        self.backbone_net = Pointnet_Backbone(
            input_channels, use_xyz,
            sample_method=sample_method,
            first_sample_method=first_sample_method)
        self.cosine = nn.CosineSimilarity(dim=1)
        self.mlp = pt_utils.SharedMLP([4+256,256,256,256], bn=True)
        self.FC_layer_cla = (
                pt_utils.Seq(256)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(1, activation=None))
        self.fea_layer = (pt_utils.Seq(256)
                .conv1d(256, bn=True)
                .conv1d(256, activation=None))
        self.vote_layer = (
                pt_utils.Seq(3+256)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(3 + 256, activation=None))
        self.group5 = pointnet2_utils.QueryAndGroup(1.0, 8, use_xyz=use_xyz)
        self.group3 = pointnet2_utils.QueryAndGroup(0.3, 8, use_xyz=use_xyz)
        self.group1 = pointnet2_utils.QueryAndGroup(0.1, 16, use_xyz=use_xyz)

        self.vote_aggregation = PointnetSAModule(
                radius=0.3,
                nsample=16,
                mlp=[1+256, 256, 256, 256],
                use_xyz=use_xyz,
                sample_method=vote_sample_method)
        self.num_proposal = input_size // 16 # 64
        self.FC_proposal = (
                pt_utils.Seq(256 + 1 + 256 + 3 + 256 + 3) # + 128 + 3)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(3+1+1, activation=None))

        multihead_attn = MultiheadAttention(
            feature_dim=d_model, n_head=1, key_feature_dim=128)

        if self.with_pos_embed:
            encoder_pos_embed = PositionEmbeddingLearned(3, d_model)
            decoder_pos_embed = PositionEmbeddingLearned(3, d_model)
        else:
            encoder_pos_embed = None
            decoder_pos_embed = None

        self.encoder = TransformerEncoder(
            multihead_attn=multihead_attn, FFN=None,
            d_model=d_model, num_encoder_layers=num_layers,
            self_posembed=encoder_pos_embed)
        self.decoder = TransformerDecoder(
            multihead_attn=multihead_attn, FFN=None,
            d_model=d_model, num_decoder_layers=num_layers,
            self_posembed=decoder_pos_embed)

    def xcorr(self, x_label, x_object, template_xyz):       
        B = x_object.size(0)
        f = x_object.size(1)
        n1 = x_object.size(2)
        n2 = x_label.size(2)

        # MxN
        final_out_cla = self.cosine(x_object.unsqueeze(-1).expand(B,f,n1,n2), x_label.unsqueeze(2).expand(B,f,n1,n2))
        # (1+3)xMxN
        fusion_feature = torch.cat((final_out_cla.unsqueeze(1),
                                    template_xyz.transpose(1, 2).contiguous().unsqueeze(-1).expand(B, 3, n1, n2)),
                                   dim=1)
        # (1+3+f)xMxN
        fusion_feature = torch.cat((fusion_feature, x_object.unsqueeze(-1).expand(B, f, n1, n2)), dim = 1)

        fusion_feature = self.mlp(fusion_feature)

        fusion_feature = F.max_pool2d(fusion_feature, kernel_size=[fusion_feature.size(2), 1])
        fusion_feature = fusion_feature.squeeze(2)
        fusion_feature = self.fea_layer(fusion_feature)

        return fusion_feature

    def transform_fuse(self, search_feature, search_xyz,
                       template_feature, template_xyz):
        """Use transformer to fuse feature.

        template_feature : BxCxN
        template_xyz : BxNx3
        """
        # BxCxN -> NxBxC
        search_feature = search_feature.permute(2, 0, 1)
        template_feature = template_feature.permute(2, 0, 1)

        num_img_train = search_feature.shape[0]
        num_img_template = template_feature.shape[0]

        ## encoder
        encoded_memory = self.encoder(template_feature,
            query_pos=template_xyz if self.with_pos_embed else None)

        encoded_feat = self.decoder(search_feature,
                                    memory=encoded_memory,
                                    query_pos=search_xyz)

        # NxBxC -> BxCxN
        encoded_feat = encoded_feat.permute(1, 2, 0)
        encoded_feat = self.fea_layer(encoded_feat)

        return encoded_feat

    def forward(self, input_dict):
        """
            template: B*512*3 or B*512*6
            search: B*1024*3 or B*1024*6
        """
        template = input_dict['template']
        search = input_dict['search']
        output_dict = {}

        template_output_dict = \
            self.backbone_net(template, [self.input_size // 4,
                              self.input_size // 8,
                              self.input_size // 16],
                              keep_first_half=True)
        template_features = template_output_dict['features']

        search_output_dict = \
            self.backbone_net(search, [self.input_size // 2,
                              self.input_size // 4,
                              self.input_size // 8],
                              target_features=template_features,
                              keep_first_half=False)

        # update label
        if 'cls_label' in input_dict:
            idxs = search_output_dict['idxs']
            b = idxs[0].shape[0]
            idxs_list = defaultdict(list)
            for i in range(len(idxs)):
                for bi in range(b):
                    if i == 0:
                        idxs_list[i].append(idxs[0][bi])
                    else:
                        idxs_list[i].append(idxs_list[i - 1][bi][idxs[i][bi].long()])

            final_idx = torch.stack(idxs_list[len(idxs) - 1], dim=0)
            output_dict['cls_label'] = torch.gather(
                input_dict['cls_label'], 1, final_idx.long())
            output_dict['reg_label'] = torch.gather(
                input_dict['reg_label'], 1, final_idx.long().unsqueeze(-1).repeat(1, 1, 4))

        template_xyz = template_output_dict['xyz']
        template_feature = template_output_dict['feature']

        search_xyz = search_output_dict['xyz']
        search_feature = search_output_dict['feature']
        fusion_feature = self.transform_fuse(
            search_feature, search_xyz, template_feature, template_xyz)

        estimation_cla = self.FC_layer_cla(fusion_feature)
        score = estimation_cla.sigmoid()

        fusion_xyz_feature = torch.cat(
            (search_xyz.transpose(1, 2).contiguous(), fusion_feature),
            dim=1)

        offset_feature = self.vote_layer(fusion_xyz_feature)
        offset = offset_feature[:, :3, :].transpose(1, 2).contiguous()
        fusion_feature = fusion_feature + offset_feature[:, 3:, :]
        proposals = search_output_dict['xyz'] - offset

        temp_pooling_feature1 = self.group5(
            template_output_dict['xyzs'][-1], proposals, template_output_dict['features'][-1])
        temp_pooling_feature1, _ = torch.max(temp_pooling_feature1, dim=-1, keepdim=False)

        pooling_feature = temp_pooling_feature1

        search_pooling_feature1 = self.group5(
            search_output_dict['xyzs'][-1], search_output_dict['xyz'], search_output_dict['features'][-1])
        search_pooling_feature1, _ = torch.max(search_pooling_feature1, dim=-1, keepdim=False)

        search_pooling_feature = search_pooling_feature1
        pooling_feature = torch.cat([pooling_feature, search_pooling_feature], dim=1)

        proposal_features = torch.cat([score, pooling_feature, fusion_feature], dim=1)
        proposal_offsets = self.FC_proposal(proposal_features)

        estimation_boxs = torch.cat(
            (proposal_offsets[:, 0:3, :] + offset.detach().transpose(1, 2).contiguous(),
             proposal_offsets[:, 3:5, :]),
            dim=1)

        output_dict.update({
            'estimation_cla' : estimation_cla.squeeze(1),
            'estimation_reg' : offset,
            'estimation_box' : estimation_boxs.transpose(1, 2).contiguous(),
            'center_xyz' : offset.detach() 
        })
        return output_dict

