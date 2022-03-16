import argparse
import os
import random
import time
import logging
import pdb
from tqdm import tqdm
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from Dataset import SiameseTrain
from pointnet2.models import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=160, help='number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')
parser.add_argument('--input_feature_num', type=int, default = 0,  help='number of input point features')
parser.add_argument('--data_dir', type=str, default = './data/kitti',  help='dataset path')
parser.add_argument('--category_name', type=str, default = 'Car',  help='Object to Track (Car/Pedestrian/Van/Cyclist)')
parser.add_argument('--save_root_dir', type=str, default='results',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model name for training resume')
parser.add_argument('--optimizer', type=str, default = '',  help='optimizer name for training resume')
parser.add_argument('--tiny', type=bool, default=False)
parser.add_argument('--input_size', type=int, default=1024)
parser.add_argument('--save_interval', type=int, default=1)
opt = parser.parse_args()

if opt.category_name == 'Ped':
    opt.category_name = 'Pedestrian'
if opt.category_name == 'Cyc':
    opt.category_name = 'Cyclist'
print(opt)

opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

save_dir = opt.save_root_dir
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')

# 1. Load data
train_data = SiameseTrain(
            input_size=opt.input_size,
            path= opt.data_dir,
            split='Train' if not opt.tiny else 'TinyTrain',
            category_name=opt.category_name,
            offset_BB=0.1, # opt.offset,
            scale_BB=1.0) # opt.scale)

train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    pin_memory=True)

test_data = SiameseTrain(
    input_size=opt.input_size,
    path=opt.data_dir,
    split='Valid' if not opt.tiny else 'TinyValid',
    category_name=opt.category_name,
    offset_BB=0.1, # opt.offset,
    scale_BB=1.0) # opt.scale)

test_dataloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=int(opt.batchSize // 2),
    shuffle=False,
    num_workers=int(opt.workers),
    pin_memory=True)

print('#Train data:', len(train_data), '#Test data:', len(test_data))
netR = get_model(name='T', # opt.type,
                 input_channels=opt.input_feature_num,
                 use_xyz=True,
                 input_size=opt.input_size)
netR = torch.nn.DataParallel(netR)
if opt.model != '':
    netR.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))

netR.cuda()
# print(netR)

criterion_cla = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0])).cuda()
criterion_reg = nn.MSELoss(reduction='none').cuda()
criterion_objective = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([2.0]),
    reduction='none').cuda()
criterion_box = nn.MSELoss(reduction='none').cuda()
optimizer = optim.Adam(netR.parameters(),
                       lr=opt.learning_rate,
                       betas=(0.5, 0.999),
                       eps=1e-6)
if opt.optimizer != '':
    optimizer.load_state_dict(torch.load(os.path.join(save_dir, opt.optimizer)))
scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)


def one_sample_step(input_dict, model, optimizer, train=True):
    optimizer.zero_grad()
    output_dict = model(input_dict)

    label_cla = output_dict['cls_label']
    label_reg = output_dict['reg_label']

    estimation_cla = output_dict['estimation_cla']
    estimation_reg = output_dict['estimation_reg'] # vote xyz
    estimation_box = output_dict['estimation_box']
    center_xyz = output_dict['center_xyz'] # candi

    loss_cla = criterion_cla(estimation_cla, label_cla)

    # vote -> box center
    loss_reg = criterion_reg(estimation_reg, label_reg[:, :, 0:3]) # 16x128x3
    loss_reg = (loss_reg.mean(2) * label_cla).sum() / (label_cla.sum() + 1e-06)

    K = center_xyz.shape[1]
    dist = torch.sum((center_xyz - label_reg[:, :, 0:3]) ** 2, dim=-1)
    dist = torch.sqrt(dist + 1e-6)
    B = dist.size(0)

    objectness_mask = torch.ones((B, K), requires_grad=False).float().cuda()
    objectness_label = label_cla

    box_mask = label_cla
    loss_objective = criterion_objective(estimation_box[:, :, 4], objectness_label)
    loss_objective = torch.sum(loss_objective * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)
    loss_box = criterion_box(estimation_box[:, :, 0:4], label_reg[:, 0:K, :])
    loss_box = (loss_box.mean(2) * box_mask).sum() / (box_mask.sum() + 1e-06)		
    loss = loss_cla + loss_reg + 1.0 * loss_box + 1.0 * loss_objective

    if train:
        loss.backward()
        optimizer.step()

    estimation_cla_cpu = estimation_cla.sigmoid().detach().cpu().numpy()
    label_cla_cpu = label_cla.detach().cpu().numpy()
    correct = float(np.sum((
        estimation_cla_cpu > 0.4) == label_cla_cpu)
        ) / label_cla_cpu.size
    true_correct = float(np.sum(
        (np.float32(estimation_cla_cpu > 0.4)
                    + label_cla_cpu) == 2)) \
        / np.sum(label_cla_cpu)

    return {
        'correct' : correct,
        'true_correct' : true_correct,
        'loss_cla' : loss_cla,
        'loss_reg' : loss_reg,
        'loss_box' : loss_box,
        'loss_objective' : loss_objective,
        'loss' : loss
    }


for epoch in range(opt.nepoch):
    scheduler.step(epoch)
    print('======>>>>> Online epoch: #%d, lr=%f <<<<<======' %(epoch, scheduler.get_lr()[0]))
    # 3.1 switch to train mode
    # torch.cuda.synchronize()
    netR.train()
    train_mse = 0.0
    timer = time.time()

    batch_correct = 0.0
    batch_cla_loss = 0.0
    batch_reg_loss = 0.0
    batch_box_loss = 0.0
    batch_num = 0.0
    batch_iou = 0.0
    batch_true_correct = 0.0
    for i, input_dict in enumerate(tqdm(train_dataloader, 0)):
        if len(input_dict['search']) == 1:
            continue
        # torch.cuda.synchronize()
        # 3.1.1 load inputs and targets
        for k, v in input_dict.items():
            input_dict[k] = Variable(v, requires_grad=False).cuda()
        output_dict = one_sample_step(input_dict, netR, optimizer)

        correct = output_dict['correct']
        true_correct = output_dict['true_correct']
        loss_cla = output_dict['loss_cla']
        loss_reg = output_dict['loss_reg']
        loss = output_dict['loss']

        train_mse = train_mse + loss.data * len(input_dict['search'])
        batch_correct += correct
        batch_cla_loss += loss_cla.data
        batch_reg_loss += loss_reg.data
        batch_num += 1 # len(input_dict['search'])
        batch_true_correct += true_correct
        if (i + 1) % 20 == 0:
            print('\n ---- batch: %03d ----' % (i+1))
            print('cla_loss: %f, reg_loss: %f, box_loss: %f' %
                   (batch_cla_loss/20,batch_reg_loss/20,batch_box_loss/20))
            print('accuracy: %f' % (batch_correct / float(batch_num)))
            print('true accuracy: %f' % (batch_true_correct / float(batch_num)))
            batch_correct = 0.0
            batch_cla_loss = 0.0
            batch_reg_loss = 0.0
            batch_box_loss = 0.0
            batch_num = 0.0
            batch_true_correct = 0.0

    # time taken
    train_mse = train_mse / len(train_data)
    timer = time.time() - timer
    timer = timer / len(train_data)
    print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

    if epoch and (epoch % opt.save_interval == 0 or epoch == opt.nepoch-1):
        torch.save(netR.state_dict(), '%s/netR_%d.pth' % (save_dir, epoch))

    # 3.2 switch to evaluate mode
    netR.eval()
    test_cla_loss = 0.0
    test_reg_loss = 0.0
    test_box_loss = 0.0
    test_correct = 0.0
    test_true_correct = 0.0
    timer = time.time()
    for i, data in enumerate(tqdm(test_dataloader, 0)):
        for k, v in input_dict.items():
            input_dict[k] = Variable(v, requires_grad=False).cuda()

        with torch.no_grad():
            output_dict = one_sample_step(input_dict, netR, optimizer, train=False)

        correct = output_dict['correct']
        true_correct = output_dict['true_correct']

        test_correct += correct
        test_true_correct += true_correct

    # time taken
    timer = time.time() - timer
    timer = timer / len(test_data)
    print('==> time to learn 1 sample = %f (ms)' %(timer*1000))
    # print mse
    test_cla_loss = test_cla_loss / len(test_data)
    test_reg_loss = test_reg_loss / len(test_data)
    test_box_loss = test_box_loss / len(test_data)
    print('cla_loss: %f, reg_loss: %f, box_loss: %f, #test_data = %d' %
        (test_cla_loss, test_reg_loss, test_box_loss, len(test_data)))
    test_correct = test_correct / len(test_dataloader)
    print('mean-correct of 1 sample: %f, #test_data = %d' %(test_correct, len(test_data)))
    test_true_correct = test_true_correct / len(test_dataloader)
    print('true correct of 1 sample: %f' %(test_true_correct))
    # log
    logging.info('Epoch#%d: train error=%e, test error=%e,%e,%e, test correct=%e, %e, lr = %f' %
                (epoch, train_mse, test_cla_loss, test_reg_loss, test_box_loss,
                 test_correct, test_true_correct, scheduler.get_lr()[0]))
