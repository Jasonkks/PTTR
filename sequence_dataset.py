

import numpy as np
from collections import defaultdict

from searchspace import KalmanFiltering
from .Dataset import SiameseTrain, SiameseTest


class SequenceTrain(SiameseTrain):
    def __init__(self, seq_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len

    def get_pc_and_bbox(self, index, random_offset=True):
        this_anno = self.list_of_anno[anno_idx]
        this_PC, this_BB = self.getPCandBBfromIndex(anno_idx)
        if random_offset:
            gaussian = KalmanFiltering(bnd=[1, 1, 5])
            sample_offsets = gaussian.sample(1)[0]
        else:
            sample_offsets = np.zeros(3)
        sample_BB = utils.getOffsetBB(this_BB, sample_offsets)
        sample_PC, sample_label, sample_reg = utils.cropAndCenterPC_label(
            this_PC, sample_BB, this_BB, sample_offsets,
            offset=self.offset_BB, scale=self.scale_BB)
        
        sample_PC, sample_label, sample_reg = utils.regularizePCwithlabel(
            sample_PC, sample_label, sample_reg, self.input_size)
        return sample_PC, sample_label, sample_reg

    def getitem(self, index):
        anno_idx = self.getAnnotationIndex(index)
        this_anno = self.list_of_anno[anno_idx]
        model_idx = this_anno['model_idx']

        if this_anno["relative_idx"] == 0:
            prev_idx = 0
            fir_idx = 0
        else:
            prev_idx = anno_idx - 1
            fir_idx = anno_idx - this_anno["relative_idx"]
        
        obj_idxs = []
        while True:
            obj_idxs.append(fir_idx)

            fir_idx += 1
            if self.list_of_anno[self.getAnnotationIndex(index)]['model_idx'] != model_idx:
                break

        if self.is_training:
            if len(obj_idxs) > self.seq_len:
                random_start_idx = np.random.randint(len(obj_idxs) - self.seq_len)
                obj_idxs = obj_idxs[random_start_idx : random_start_idx + self.seq_len]
            
            if len(obj_idxs) < self.seq_len:
                obj_idxs += [-1 for _ in range(self.seq_len - len(obj_idxs))]

        ret = defaultdict(list)
        for i, idx in enumerate(obj_idxs):
            if idx >= 0:
                pc, cls_label, reg_label = self.get_pc_and_bbox(
                    idx, random_offset=self.is_training)

                ret['points'].append(pc)
                ret['cls_label'].append(cls_label)
                ret['reg_label'].append(reg_label)
                ret['mask'].append(1)
            else:
                ret['points'].append(np.zeros_like(ret['points'][0]))
                ret['cls_label'].append(np.zeros_like(ret['cls_label'][0]))
                ret['reg_label'].append(np.zeros_like(ret['reg_label'][0]))
                ret['mask'].append(0)

        for k, v in ret.items():
            ret[k] = np.stack(v, dim=0)

        return ret

    def __len__(self):
        return len(self.list_of_tracklet_anno)
