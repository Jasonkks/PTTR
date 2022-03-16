# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.

import os
import pickle
import copy
import numpy as np
import torch
import multiprocessing
import SharedArray
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        if self.training:
            self.data_ratio = self.dataset_cfg.get('DATA_RATIO', 1)
        else:
            self.data_ratio = self.dataset_cfg.get('TEST_DATA_RATIO', 1)
        if self.data_ratio < 1:
            sequence_length = max(int(len(self.sample_sequence_list) * self.data_ratio), 1)
            self.sample_sequence_list = self.sample_sequence_list[:sequence_length]
            self.logger.info('subsampled sequence number to {}'.format(sequence_length))

        self.infos = []
        self.include_waymo_data(self.mode)

        self.use_shared_memory = self.dataset_cfg.get('USE_SHARED_MEMORY', False) and self.training
        if self.use_shared_memory:
            self.shared_memory_file_limit = self.dataset_cfg.get('SHARED_MEMORY_FILE_LIMIT', 0x7FFFFFFF)
            self.load_data_to_shared_memory()

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.include_waymo_data(self.mode)

    def include_waymo_data(self, mode):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = self.data_path / sequence_name / ('%s.pkl' % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                print('info path not exist:', info_path)
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))

    def load_data_to_shared_memory(self):
        self.logger.info(f'Loading training data to shared memory (file limit={self.shared_memory_file_limit})')

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = self.infos[:self.shared_memory_file_limit] \
            if self.shared_memory_file_limit < len(self.infos) else self.infos
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            sa_key = f'{sequence_name}___{sample_idx}'
            if os.path.exists(f"/dev/shm/{sa_key}"):
                continue

            points = self.get_lidar(sequence_name, sample_idx)
            common_utils.sa_create(f"shm://{sa_key}", points)

        dist.barrier()
        self.logger.info('Training data has been saved to shared memory')

    def clean_shared_memory(self):
        self.logger.info(f'Clean training data from shared memory (file limit={self.shared_memory_file_limit})')

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = self.infos[:self.shared_memory_file_limit] \
            if self.shared_memory_file_limit < len(self.infos) else self.infos
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            sa_key = f'{sequence_name}___{sample_idx}'
            if not os.path.exists(f"/dev/shm/{sa_key}"):
                continue

            SharedArray.delete(f"shm://{sa_key}")

        if num_gpus > 1:
            dist.barrier()
        self.logger.info('Training data has been deleted from shared memory')

    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if not sequence_file.exists():
            found_sequence_file = sequence_file
            for pre_text in ['training', 'validation', 'testing']:
                if not sequence_file.exists():
                    temp_sequence_file = Path(str(sequence_file).replace('segment', pre_text + '_segment'))
                    if temp_sequence_file.exists():
                        found_sequence_file = temp_sequence_file
                        break
            if not found_sequence_file.exists():
                found_sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))
            if found_sequence_file.exists():
                sequence_file = found_sequence_file
        return sequence_file

    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1):
        from functools import partial
        from . import waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            waymo_utils.process_single_sequence,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label
        )
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        with multiprocessing.Pool(num_workers) as p:
            sequence_infos = list(tqdm(p.imap(process_single_sequence, sample_sequence_file_list),
                                       total=len(sample_sequence_file_list)))

        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos

    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        if not self.dataset_cfg.get('DISABLE_NLZ_FLAG_ON_POINTS', False):
            points_all = points_all[NLZ_flag == -1]
        points_all[:, 3] = np.tanh(points_all[:, 3])
        return points_all

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']

        if self.use_shared_memory and index < self.shared_memory_file_limit:
            sa_key = f'{sequence_name}___{sample_idx}'
            points = SharedArray.attach(f"shm://{sa_key}").copy()
        else:
            points = self.get_lidar(sequence_name, sample_idx)

        input_dict = {
            'points': points,
            'frame_id': info['frame_id'],
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            if self.training and self.dataset_cfg.get('FILTER_EMPTY_BOXES_FOR_TRAIN', False):
                mask = (annos['num_points_in_gt'] > 0)  # filter empty boxes
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])
        data_dict.pop('num_points_in_gt', None)
        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def create_groundtruth_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=10,
                                    processed_data_tag=None):
        database_save_path = save_path / ('%s_gt_database_%s_sampled_%d' % (processed_data_tag, split, sampled_interval))
        db_info_save_path = save_path / ('%s_waymo_dbinfos_%s_sampled_%d.pkl' % (processed_data_tag, split, sampled_interval))
        db_data_save_path = save_path / ('%s_gt_database_%s_sampled_%d_global.npy' % (processed_data_tag, split, sampled_interval))
        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        point_offset_cnt = 0
        stacked_gt_points = []
        for k in range(0, len(infos), sampled_interval):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]

            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            points = self.get_lidar(sequence_name, sample_idx)

            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            if k % 4 != 0 and len(names) > 0:
                mask = (names == 'Vehicle')
                names = names[~mask]
                difficulty = difficulty[~mask]
                gt_boxes = gt_boxes[~mask]

            if k % 2 != 0 and len(names) > 0:
                mask = (names == 'Pedestrian')
                names = names[~mask]
                difficulty = difficulty[~mask]
                gt_boxes = gt_boxes[~mask]

            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(num_obj):
                filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                               'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                               'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}

                    # it will be used if you choose to use shared memory for gt sampling
                    stacked_gt_points.append(gt_points)
                    db_info['global_data_offset'] = [point_offset_cnt, point_offset_cnt + gt_points.shape[0]]
                    point_offset_cnt += gt_points.shape[0]

                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

        # it will be used if you choose to use shared memory for gt sampling
        stacked_gt_points = np.concatenate(stacked_gt_points, axis=0)
        np.save(db_data_save_path, stacked_gt_points)


def create_waymo_infos(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='waymo_processed_data',
                       workers=min(64, multiprocessing.cpu_count()), skip_gt_database=True):
    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, train_split))
    val_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, val_split))

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    waymo_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)

    dataset.set_split(val_split)
    waymo_infos_val = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(waymo_infos_val, f)
    print('----------------Waymo info val file is saved to %s----------------' % val_filename)
    
    if not skip_gt_database:
        print('---------------Start create groundtruth database for data augmentation---------------')
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        dataset.set_split(train_split)
        dataset.create_groundtruth_database(
            info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
            used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=processed_data_tag
        )
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    # parser.add_argument('--processed_data_tag', type=str, default='waymo_processed_data_v0_5_0', help='')
    args = parser.parse_args()

    if args.func == 'create_waymo_infos':
        import yaml
        from easydict import EasyDict
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        # dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
        args.processed_data_tag = dataset_cfg.PROCESSED_DATA_TAG
        create_waymo_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'waymo',
            save_path=ROOT_DIR / 'data' / 'waymo',
            raw_data_tag='raw_data',
            processed_data_tag=args.processed_data_tag
        )
