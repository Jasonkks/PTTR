from pcdet.datasets import WaymoDataset
from pcdet.config import cfg, cfg_from_yaml_file

import numpy as np
import os
import copy
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from functools import partial
import copy


from data_classes import PointCloud, Box
import kitty_utils as utils
from kitty_utils import getModel
from searchspace import KalmanFiltering


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


class WaymoSiameseDataset(Dataset):
    def __init__(self, input_size, path,
                 category_name='Vehicle',
                 offset_BB=0,
                 scale_BB=1.0,
                 align=True,
                 min_seq_len=2,
                 min_pts_in_gt=10):
        self.waymo_ds = WaymoTrackingDataset(
            path, class_names=[category_name],
        )

        self.offset_BB = offset_BB
        self.scale_BB = scale_BB
        self.input_size = input_size
        self.align = align
        self.tracklets = []
        for i in range(len(self.waymo_ds.tracklets)):
            self.tracklets.append(self.waymo_ds.load_sample(i))

        print("DATASET: Load %d tracklets with %d samples" % 
            (len(self.tracklets), sum(len(_) for _ in self.tracklets)))

        # filter min_pts_in_gt
        new_tracklets = []
        for trk in self.tracklets:
            new_trk = []
            for sp in trk:
                if sp['num_points_in_gt'] < min_pts_in_gt:
                    if len(new_trk):
                        new_tracklets.append(new_trk)
                        new_trk = []
                else:
                    new_trk.append(sp)
        self.tracklets = new_tracklets
        print("DATASET [After remove pt less than %d]: Load %d tracklets with %d samples" % 
            (min_pts_in_gt, len(self.tracklets), sum(len(_) for _ in self.tracklets)))

        # filter min_seq_len
        new_tracklets = []
        for trk in self.tracklets:
            if len(trk) >= min_seq_len:
                new_tracklets.append(trk)
        self.tracklets = new_tracklets
        print("DATASET [After remove tracklet less than %d]: Load %d tracklets with %d samples" % 
            (min_seq_len, len(self.tracklets), sum(len(_) for _ in self.tracklets)))

    def _transform(self, pc, T):
        assert pc.shape[1] == 3
        pc = np.concatenate([pc, np.ones((pc.shape[0], 1))], axis=1)
        transformed_pc = T @ np.transpose(pc)
        transformed_pc = np.transpose(transformed_pc)
        transformed_pc = transformed_pc[:, :3]
        return transformed_pc

    def pose_compensate(self, T0, T1, points):
        # points from pose T0 to T1
        T_trans = np.linalg.inv(T1) @ T0
        pts = self._transform(np.copy(points[:, :3]), T_trans)
        return np.concatenate((pts, points[:, 3:]), axis=1)

    def _getPC(self, pc):
        return PointCloud(pc.T)

    def _getBB(self, box):
        center = [box[0], box[1], box[2]]
        size = [box[4], box[3], box[5]]
        orientation = Quaternion(
            axis=[0, 0, 1], radians=box[6])
        return Box(center, size, orientation)

    def __getitem__(self, index):
        return self.getitem(index)

    def getitem(self, index, sample_idx=None, return_raw=False):
        sample = self.tracklets[index]

        def random_box(box, center_offset, w_ratio, h_ratio, flag):
            if not flag:
                return box
            box = copy.deepcopy(box)
            box.center[0] += center_offset[0] * box.wlh[1]
            box.center[1] += center_offset[1] * box.wlh[0]
            box.wlh[0] *= w_ratio
            box.wlh[1] *= h_ratio
            return box

        random_box_func = partial(random_box, **dict(
            center_offset=[np.random.uniform(-0.4, 0.4),
                           np.random.uniform(-0.4, 0.4)],
            w_ratio=np.random.uniform(0.3, 1.0),
            h_ratio=np.random.uniform(0.3, 1.0),
            flag=np.random.uniform() < 0.0 # prob
        ))

        if sample_idx is None:
            sample_idx = np.random.randint(len(sample))
        current_sample = sample[sample_idx]

        this_PC = self.waymo_ds.get_lidar(
            current_sample['sequence_name'],
            current_sample['sample_idx'])
        pose_this = current_sample['pose']
        this_BB = current_sample['box']

        this_PC = self._getPC(this_PC)
        this_BB = self._getBB(this_BB)

        if return_raw:
            return {'pc' : this_PC, 'box' : this_BB, 'pose' : pose_this}

        if sample_idx == 0:
            sample_offsets = np.zeros(4)
        else:
            gaussian = KalmanFiltering(bnd=[1, 1, 1, 1])
            sample_offsets = gaussian.sample(1)[0]
            sample_offsets[1] /= 2.0
            sample_offsets[0] *= 2

        # Random bbox
        sample_BB = utils.getOffsetBB(this_BB, sample_offsets)
        sample_BB = random_box_func(box=sample_BB)

        sample_PC, sample_label, sample_reg = utils.cropAndCenterPC_label(
            this_PC, sample_BB, this_BB, sample_offsets,
            offset=self.offset_BB, scale=self.scale_BB)

        if sample_PC.nbr_points() <= 10:
            return self.getitem(np.random.randint(0, self.__len__()))

        random_downsample = np.random.uniform() < 0.0
        def _random_sample_pts(pc, num):
            p = np.array(pc.points, dtype=np.float32)
            if p.shape[1] < 10:
                return pc
            new_idx = np.random.randint(low=0, high=p.shape[1], size=num, dtype=np.int64)
            p = p[:, new_idx]
            pc.points = p
            return pc
        if random_downsample:
            random_downsample_pc_func = partial(_random_sample_pts,
                num=np.random.randint(min(128, sample_PC.points.shape[1] - 1),
                                      sample_PC.points.shape[1]))
            sample_PC = random_downsample_pc_func(sample_PC)

        sample_PC, sample_label, sample_reg = utils.regularizePCwithlabel(
            sample_PC, sample_label, sample_reg, self.input_size)

        if current_sample["sample_idx"] == 0:
            prev_idx = 0
            fir_idx = 0
        else:
            prev_idx = sample_idx - 1
            fir_idx = 0
        gt_PC_pre = self.waymo_ds.get_lidar(
            sample[prev_idx]['sequence_name'],
            sample[prev_idx]['sample_idx'])
        pose_prev = sample[prev_idx]['pose']
        gt_BB_pre = sample[prev_idx]['box']
        gt_PC_pre = self._getPC(gt_PC_pre)
        gt_BB_pre = self._getBB(gt_BB_pre)

        gt_BB_pre = random_box_func(box=gt_BB_pre)

        if sample_idx == 0:
            samplegt_offsets = np.zeros(4)
        else:
            samplegt_offsets = np.random.uniform(low=-0.3, high=0.3, size=4)
            samplegt_offsets[0] *= 2.0
        gt_BB_pre = utils.getOffsetBB(gt_BB_pre, samplegt_offsets)
        gt_PC = getModel([gt_PC_pre], [gt_BB_pre], offset=self.offset_BB, scale=self.scale_BB)
        if random_downsample:
            gt_PC = random_downsample_pc_func(gt_PC)

        if gt_PC.nbr_points() <= 20:
            return self.getitem(np.random.randint(0, len(self)))

        gt_PC = utils.regularizePC(gt_PC, self.input_size)

        ret = {
            'search' : sample_PC,
            'template' : gt_PC,
            'cls_label' : sample_label, # whether in box
            'reg_label' : sample_reg # box
        }
        return ret # sample_PC, sample_label, sample_reg, gt_PC


    def __len__(self):
        return len(self.tracklets)

class WaymoTrackingDataset(WaymoDataset):
    """
    """
    def __init__(self, output_dir,
                 class_names=['Vehicle',], # Pedestrian Cyclist
                 training=True,
                 root_path=Path('./OpenPCDet/data/waymo/'),
                 logger=logging,
                 data_ratio=0.1,
                 ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        cfg_from_yaml_file('./waymo_dataset.yaml', cfg)
        cfg.DATA_PATH = root_path
        cfg.DATA_RATIO = data_ratio
        cfg.TEST_DATA_RATIO = data_ratio

        super(WaymoTrackingDataset, self).__init__(
              dataset_cfg=cfg,
              class_names=class_names,
              training=training,
              root_path=root_path,
              logger=logger)
        self.output_dir = output_dir
        self.class_name = class_names[0]
        assert len(class_names) == 1

        self.tracklets = [tmp for tmp in os.listdir(self.output_dir) if tmp.startswith(self.class_name)]

    def preprocess(self):
        tracklet = {}
        tracklet_set = set()
        for i in range(len(self)):
            new_tracklet = {}
            data_dict = self[i]

            num_obj = len(data_dict['gt_boxes'])

            for obj_idx in range(num_obj):
                box = data_dict['gt_boxes'][obj_idx]
                id = data_dict['obj_ids'][obj_idx]
                num_pt_in_gt = data_dict['num_points_in_gt'][obj_idx]

                sample = {
                    'id' : id,
                    'box' : box,
                    # 'points' : data_dict['points'],
                    'frame_id' : data_dict['frame_id'],
                    'pose' : data_dict['pose'],
                    'sequence_name' : data_dict['sequence_name'],
                    'sample_idx' : data_dict['sample_idx'],
                    'num_points_in_gt' : num_pt_in_gt
                }

                if id in tracklet:
                    new_tracklet[id] = tracklet[id]
                    assert sample['sample_idx'] == tracklet[id][-1]['sample_idx'] + 1
                    new_tracklet[id].append(sample)
                else:
                    new_tracklet[id] = [sample]

            for id in tracklet:
                if id in new_tracklet:
                    continue
                
                # treat as new obj
                count = 0
                new_id = id
                while new_id in tracklet_set:
                    new_id = '%s_%d' % (id, count)
                ori_id = id
                id = new_id

                with open(os.path.join(self.output_dir, '%s_%d_' % (self.class_name, len(tracklet_set)) + id + '.npy'), 'wb') as f:
                    logging.info('%d save %s with length %d' % (len(tracklet_set), id, len(tracklet[ori_id])))
                    np.save(f, tracklet[ori_id], allow_pickle=True)

                tracklet_set.add(id)
            tracklet = new_tracklet

        # save all in new tracklet
        for trk_id, trk_sample in tracklet.items():
            # rename
            new_id = trk_id
            count = 0
            while new_id in tracklet_set:
                new_id = '%s_%d' % (trk_id, count)

            with open(os.path.join(
                    self.output_dir, '%s_%d_' % (self.class_name, len(tracklet_set)) + new_id + '.npy'),
                    'wb') as f:
                logging.info('%d save %s with length %d' % (len(tracklet_set), new_id, len(tracklet[trk_id])))
                np.save(f, trk_sample, allow_pickle=True)
                assert not new_id in tracklet_set, '%s already exists' % new_id 
                tracklet_set.add(new_id)

    def load_sample(self, tracklet_id):
        track_id = self.tracklets[tracklet_id]
        with open(os.path.join(self.output_dir, track_id), 'rb') as f:
            sample_load = np.load(f, allow_pickle=True)
        return sample_load

    def __getitem__(self, index):
        data_dict = self._get_data_dict(index)
        return data_dict

    def _get_data_dict(self, index):
        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']

        input_dict = {
            'frame_id': info['frame_id'],
            'pose' : info['pose'],
            'sample_idx' : sample_idx,
            'sequence_name' : sequence_name,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = drop_info_with_name(annos, name='unknown')
            gt_boxes_lidar = annos['gt_boxes_lidar']
            num_points_in_gt = annos['num_points_in_gt']

            name = annos['name']
            obj_ids = annos['obj_ids']

            annos.update({
                'num_points_in_gt' : num_points_in_gt,
                'name' : name,
                'obj_ids' : obj_ids,
                'gt_boxes_lidar' : gt_boxes_lidar
            })

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'obj_ids' : annos['obj_ids'],
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = input_dict

        if data_dict.get('gt_boxes', None) is not None:
            selected = keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            data_dict['obj_ids'] = data_dict['obj_ids'][selected]

        return data_dict


if __name__ == '__main__':
    print(__file__)

    for category, ratio in zip(['Vehicle', 'Pedestrian', 'Cyclist'], [0.1, 0.2, 1.0]):
        ds = WaymoTrackingDataset('./data/waymo_tracking/train/', training=True, class_names=[category], data_ratio=ratio)
        ds.preprocess()

        ds = WaymoTrackingDataset('./data/waymo_tracking/test/', training=False, class_names=[category], data_ratio=ratio)
        ds.preprocess()


