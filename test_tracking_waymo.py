import time
import os
import logging
import argparse
import random
from pyquaternion import Quaternion

import numpy as np
from tqdm import tqdm

import torch

import kitty_utils as utils
import copy
from datetime import datetime


from metrics import AverageMeter, Success, Precision
from metrics import estimateOverlap, estimateAccuracy, estimateIOU3d
from waymo_dataset import WaymoSiameseDataset

import torch.nn.functional as F
from torch.autograd import Variable

from pointnet2.models import get_model


def _transform(self, pc, T):
    assert pc.shape[1] == 3
    pc = np.concatenate([pc, np.ones((pc.shape[0], 1))], axis=1)
    transformed_pc = T @ np.transpose(pc)
    transformed_pc = np.transpose(transformed_pc)
    transformed_pc = transformed_pc[:, :3]
    return transformed_pc

def pose_compensate(self, T0, T1, points):
    T_trans = np.linalg.inv(T1) @ T0
    pts = self._transform(np.copy(points[:, :3]), T_trans)
    return np.concatenate((pts, points[:, 3:]), axis=1)


def test(loader, model, epoch=-1,
         shape_aggregation="",
         reference_BB="",
         model_fusion="pointcloud",
         max_iter=-1,
         IoU_Space=3,
         align=True):
    """


    """
    dataset = loader
    batch_time = AverageMeter()
    data_time = AverageMeter()

    Success_main = Success()
    Precision_main = Precision()
    Success_batch = Success()
    Precision_batch = Precision()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    batch_num = 0

    for tracklet_idx in range(len(loader)):          
        batch_num = batch_num+1
        data_time.update((time.time() - end))

        seq_len = len(loader.tracklets[tracklet_idx])
        samples = [loader.getitem(tracklet_idx, sample_idx, return_raw=True) for sample_idx in range(seq_len)]
        PCs = [s['pc'] for s in samples]
        BBs = [s['box'] for s in samples]
        poses = [s['pose'] for s in samples]

        # for sample_idx in range(seq_len):
        results_BBs = []
        for i, _ in enumerate(PCs):
            # this_anno = list_of_anno[i]
            this_BB = BBs[i]
            this_PC = PCs[i]
            pose_this = poses[i]
            gt_boxs = []
            result_boxs = []

            # INITIAL FRAME
            if i == 0:
                box = BBs[i]
                results_BBs.append(box)
                model_PC = utils.getModel([this_PC], [this_BB],
                                          offset=dataset.offset_BB,
                                          scale=dataset.scale_BB)
            else:
                previous_BB = BBs[i - 1]

                # DEFINE REFERENCE BB
                if ("previous_result".upper() in reference_BB.upper()):
                    ref_BB = results_BBs[-1]
                    pose_prev = poses[i - 1]
                elif ("previous_gt".upper() in reference_BB.upper()):
                    ref_BB = previous_BB
                    pose_prev = poses[i - 1]
                    # ref_BB = utils.getOffsetBB(this_BB,np.array([-1,1,1]))
                elif ("current_gt".upper() in reference_BB.upper()):
                    ref_BB = this_BB
                    pose_prev = poses[i]

                if align:
                    ref_BB = copy.deepcopy(ref_BB)
                    # ref_BB.center = pose_compensate(pose_prev, pose_this, ref_BB.center[None, ...])[0]
                    ref_BB.transform(np.linalg.inv(pose_this) @ pose_prev)

                candidate_PC, candidate_label, candidate_reg, new_ref_box, new_this_box = \
                    utils.cropAndCenterPC_label_test(this_PC,
                                                     ref_BB, this_BB,
                                                     offset=dataset.offset_BB,
                                                     scale=dataset.scale_BB)
                candidate_PCs, candidate_labels, candidate_reg = utils.regularizePCwithlabel(
                    candidate_PC, candidate_label,candidate_reg,
                    dataset.input_size, istrain=False, keep_first_half=False)
                candidate_PCs_torch = candidate_PCs.unsqueeze(0).cuda()

                # AGGREGATION: IO vs ONLY0 vs ONLYI vs ALL
                if ("firstandprevious".upper() in shape_aggregation.upper()):
                    model_PC = utils.getModel(
                        [PCs[0], PCs[i-1]],
                        [results_BBs[0], results_BBs[i-1]],
                        offset=dataset.offset_BB,
                        scale=dataset.scale_BB)
                elif ("first".upper() in shape_aggregation.upper()):
                    model_PC = utils.getModel(
                        [PCs[0]], [results_BBs[0]],
                        offset=dataset.offset_BB,
                        scale=dataset.scale_BB)
                elif ("previous".upper() in shape_aggregation.upper()):
                    model_PC = utils.getModel(
                        [PCs[i-1]], [results_BBs[i-1]],
                        offset=dataset.offset_BB,
                        scale=dataset.scale_BB)
                elif ("all".upper() in shape_aggregation.upper()):
                    model_PC = utils.getModel(
                        PCs[:i], results_BBs,
                        offset=dataset.offset_BB,
                        scale=dataset.scale_BB)
                else:
                    model_PC = utils.getModel(
                        PCs[:i], results_BBs,
                        offset=dataset.offset_BB,
                        scale=dataset.scale_BB)

                model_PC_torch = utils.regularizePC(
                    model_PC, dataset.input_size,
                    istrain=False, keep_first_half=True).unsqueeze(0)
                model_PC_torch = Variable(
                    model_PC_torch, requires_grad=False).cuda()
                candidate_PCs_torch = Variable(
                    candidate_PCs_torch, requires_grad=False).cuda()

                input_dict = {
                    'template' : model_PC_torch,
                    'search' : candidate_PCs_torch
                }
                output_dict = model(input_dict)
                estimation_box = output_dict['estimation_box']
                estimation_boxs_cpu = estimation_box.squeeze(0).detach().cpu().numpy()
                box_idx = estimation_boxs_cpu[:,4].argmax()
                estimation_box_cpu = estimation_boxs_cpu[box_idx, 0:4]

                box = utils.getOffsetBB(ref_BB, estimation_box_cpu, training=False)
                results_BBs.append(box)

            # estimate overlap/accuracy for current sample
            this_overlap = estimateIOU3d(BBs[i], results_BBs[-1]) # , estimation_box_cpu[-1])
            this_accuracy = estimateAccuracy(BBs[i], results_BBs[-1], dim=IoU_Space)


            Success_main.add_overlap(this_overlap)
            Precision_main.add_accuracy(this_accuracy)
            Success_batch.add_overlap(this_overlap)
            Precision_batch.add_accuracy(this_accuracy)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('Test {}: '.format('%d/%d' % (tracklet_idx, len(loader)))+
              'Time {:.3f}s '.format(batch_time.avg)+
              '(it:{:.3f}s) '.format(batch_time.val)+
              'Data:{:.3f}s '.format(data_time.avg)+
              '(it:{:.3f}s), '.format(data_time.val)+
              'Succ/Prec:'+
              '{:.1f}/'.format(Success_main.average)+
              '{:.1f}'.format(Precision_main.average))
        Success_batch.reset()
        Precision_batch.reset()

    return Success_main.average, Precision_main.average



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root_dir', type=str, default='./results',  help='output folder')
    parser.add_argument('--data_dir', type=str, default = './data/waymo_tracking',  help='dataset path')
    parser.add_argument('--model', type=str, default = 'netR_59.pth',  help='model name for training resume')
    parser.add_argument('--category_name', type=str, default = 'Vehicle',  help='Object to Track (Vehicle/Pedestrian/Cyclist)')
    parser.add_argument('--shape_aggregation',required=False,type=str,default="previous",help='Aggregation of shapes (first/previous/firstandprevious/all)')
    parser.add_argument('--reference_BB',required=False,type=str,default="previous_result",help='previous_result/previous_gt/current_gt')
    parser.add_argument('--model_fusion',required=False,type=str,default="pointcloud",help='early or late fusion (pointcloud/latent/space)')
    parser.add_argument('--IoU_Space',required=False,type=int,default=3,help='IoUBox vs IoUBEV (2 vs 3)')
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--align', action='store_true')
    args = parser.parse_args()
    print (args)
    opt = args

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(args.save_root_dir, datetime.now().strftime('%Y-%m-%d %H-%M-%S.log')), level=logging.INFO)
    logging.info('======================================================')

    args.manualSeed = 1
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    netR = get_model(name='T', # args.type,
                     input_channels=0,
                     use_xyz=True,
                     input_size=args.input_size)
    netR = torch.nn.DataParallel(netR)
    if args.model != '':
        netR.load_state_dict(torch.load(os.path.join(args.save_root_dir, args.model)), strict=False)    
    netR.cuda()
    torch.cuda.synchronize()

    test_data = WaymoSiameseDataset(
        input_size=args.input_size,
        path=os.path.join(args.data_dir, 'test'),
        category_name=opt.category_name,
        offset_BB=0.1,
        scale_BB=1.0)

    Success_run = AverageMeter()
    Precision_run = AverageMeter()

    max_epoch = 1

    for epoch in range(max_epoch):
        Succ, Prec = test(
            test_data,
            netR,
            epoch=epoch + 1,
            shape_aggregation=args.shape_aggregation,
            reference_BB=args.reference_BB,
            model_fusion=args.model_fusion,
            IoU_Space=args.IoU_Space,
            align=args.align)
        Success_run.update(Succ)
        Precision_run.update(Prec)
        logging.info("mean Succ/Prec {}/{}".format(Success_run.avg, Precision_run.avg))

    print("mean Succ/Prec {}/{}".format(Success_run.avg, Precision_run.avg))

