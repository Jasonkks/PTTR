from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import etw_pytorch_utils as pt_utils

from pointnet2.utils import pointnet2_utils


if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


def calc_square_dist(a, b, return_cos=False):
    """
    Calculating square distance between a and b
    a: [bs, c, n]
    b: [bs, c, m]
    """
    a = a.transpose(1, 2)
    b = b.transpose(1, 2)
    n = a.shape[1]
    m = b.shape[1]
    num_channel = a.shape[-1]
    a_square = a.unsqueeze(dim=2)  # [bs, n, 1, c]
    b_square = b.unsqueeze(dim=1)  # [bs, 1, m, c]
    a_square = torch.sum(a_square * a_square, dim=-1)  # [bs, n, 1]
    b_square = torch.sum(b_square * b_square, dim=-1)  # [bs, 1, m]
    a_square = a_square.repeat((1, 1, m))  # [bs, n, m]
    b_square = b_square.repeat((1, n, 1))  # [bs, n, m]

    coor = torch.matmul(a, b.transpose(1, 2))  # [bs, n, m]

    if not return_cos:
        dist = a_square + b_square - 2.0 * coor  # [bs, npoint, ndataset]
    else:
        dist = coor / torch.sqrt(a_square) / torch.sqrt(b_square)
    return dist


class _PointnetSAModuleBase(nn.Module):
    def __init__(self, sample_method=None):
        super(_PointnetSAModuleBase, self).__init__()
        self.groupers = None
        self.mlps = None
        self.sample_method = sample_method

    def forward(self, xyz, features, npoint,
                target_feature=None,
                keep_first_half=False):
        # type: (_PointnetSAModuleBase, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """
        self.npoint = npoint
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()

        if target_feature is not None:
            assert features is not None
            with torch.no_grad():
                assert target_feature.shape[1] == features.shape[1]
                _, _, m = target_feature.shape
                sample_ratio = 2

                npoint = self.npoint - self.npoint // sample_ratio
                features_for_fps_distance = calc_square_dist(
                    features[..., npoint: ],
                    target_feature)

                min_val, target_idx = torch.min(features_for_fps_distance, dim=2) # bs, n
                min_val, min_top_k = torch.topk(min_val, self.npoint // sample_ratio, largest=False, dim=1)
                target_idx = npoint + min_top_k.int()
        else:
            npoint = self.npoint

        if self.sample_method is None:
            if keep_first_half:
                idx0 =  torch.arange(npoint // 2).repeat(
                    xyz.size(0), 1).int().cuda()
                idx1 =  torch.arange(npoint - npoint // 2).repeat(
                    xyz.size(0), 1).int().cuda() + xyz.shape[1] // 2
                idx = torch.cat([idx0, idx1], dim=1)
            else:
                idx = torch.arange(npoint).repeat(xyz.size(0), 1).int().cuda()
        elif self.sample_method == 'fps':
            idx = pointnet2_utils.furthest_point_sample(xyz, npoint)
        else:
            raise ValueError()

        if target_feature is not None:
            idx = torch.cat([idx, target_idx], dim=-1)

        new_xyz = (
            pointnet2_utils.gather_operation(xyz_flipped, idx)
            .transpose(1, 2)
            .contiguous())

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), idx


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, radii, nsamples, mlps, bn=True,
                 use_xyz=True, vote=False, sample_method=None):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__(sample_method=sample_method)
        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if vote is False:
                self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            else:
                self.groupers.append(
                pointnet2_utils.QueryAndGroup_score(radius, nsample, use_xyz=use_xyz))

            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, radius=None, nsample=None,
        bn=True, use_xyz=True, sample_method=None
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
            sample_method=sample_method,
        )


class PointnetProposalModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, radius=None, nsample=None, bn=True,
        use_xyz=True, vote=True, sample_method=None
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetProposalModule, self).__init__(
            mlps=[mlp],
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
            vote = vote,
            sample_method=sample_method
        )


    def forward(self, xyz, features, npoint, score):
        # type: (_PointnetSAModuleBase, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        self.npoint = npoint
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        
        if self.sample_method is None:
            new_xyz = (
                pointnet2_utils.gather_operation(
                    xyz_flipped,torch.arange(self.npoint).repeat(xyz.size(0),1).int().cuda()
                )
                .transpose(1, 2)
                .contiguous())
        else:
            new_xyz = (
                pointnet2_utils.gather_operation(
                    xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
                )
                .transpose(1, 2)
                .contiguous())

        for i in range(len(self.groupers)):
            new_features, score_id = self.groupers[i](
                xyz, new_xyz, score, features
            )  # (B, C, npoint, nsample)
            idx = torch.arange(new_features.size(0))
            new_features = new_features[idx,:,score_id,:]
            #B*C*nsample
            new_features = new_features.unsqueeze(2)
            #B*C*1*nsample
            new_xyz = new_xyz[idx,score_id,:]
            #B*3

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1).squeeze(-1)  # (B, mlp[-1])

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    from torch.autograd import Variable

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=True)
    xyz_feats = Variable(torch.randn(2, 9, 6).cuda(), requires_grad=True)

    test_module = PointnetSAModuleMSG(
        npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]]
    )
    test_module.cuda()
    print(test_module(xyz, xyz_feats))

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
        print(new_features)
        print(xyz.grad)
