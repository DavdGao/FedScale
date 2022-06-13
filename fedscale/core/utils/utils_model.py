# -*- coding: utf-8 -*-

import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import logging

# libs from fedscale
from fedscale.core.arg_parser import args
from fedscale.core.utils.nlp import mask_tokens

if args.task == "detection":
    import os
    import sys
    from torch.autograd import Variable
    import torch.nn as nn
    import torch.optim as optim
    import pickle
    from utils.rcnn.lib.roi_data_layer.roidb import combined_roidb
    from utils.rcnn.lib.roi_data_layer.roibatchLoader import roibatchLoader
    from utils.rcnn.lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
    from utils.rcnn.lib.model.rpn.bbox_transform import clip_boxes
    from utils.rcnn.lib.model.roi_layers import nms
    from utils.rcnn.lib.datasets.pascal_voc import readClass
    from utils.rcnn.lib.model.rpn.bbox_transform import bbox_transform_inv
    import numpy as np
    from utils.rcnn.lib.model.faster_rcnn.resnet import resnet

class MySGD(optim.SGD):

    def __init__(self, params, lr=0.01, momentum=0.0,
                 dampening=0, weight_decay=0, nesterov=False):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # print('Previous: {}, lr: {}, grad: {}'.format(p.data, group['lr'], d_p))
                p.data.add_(-group['lr'], d_p)
                # print('Now: {}'.format(p.data))

        return loss

    def get_delta_w(self, nestedLr=0.01):
        delta_ws = []
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if nestedLr == 0.01:
                    delta_ws.append(group['lr'] * d_p)
                else:
                    delta_ws.append(nestedLr * d_p)

        return delta_ws

def cal_accuracy(targets, outputs):
    temp_acc = 0
    temp_all_or_false = 0

    temp_len = 0

    for idx, sample in enumerate(targets):
        flag = True
        for item in outputs[idx]:
            if item in sample:
                temp_acc += 1
            else:
                flag = False

        if flag:
            temp_all_or_false += 1

        temp_len += len(sample)

    temp_all_or_false = (temp_all_or_false/float(len(targets)) * temp_len)

    return temp_acc, temp_all_or_false, temp_len

def my_test_model(model, test_datas, device='cpu', criterion=nn.NLLLoss()):
    model = model.to(device=device)
    model.eval()

    test_loss_total, correct_total, top_5_total, test_len_total = 0., 0., 0., 0.
    with torch.no_grad():
        for data_id, test_data in enumerate(test_datas):
            client_id = data_id + 1
            # client-wise metric
            test_loss, correct, top_5, test_len = 0., 0., 0., 0.
            for data, target in test_data:
                try:
                    data, target = Variable(data).to(device=device), Variable(target).to(device=device)

                    output = model(data)
                    loss = criterion(output, target)

                    test_loss += loss.data.item()  # Variable.data
                    acc = accuracy(output, target, topk=(1, 5))

                    correct += acc[0].item()
                    top_5 += acc[1].item()

                except Exception as ex:
                    logging.info(f"Testing of failed as {ex}")
                    break
                test_len += len(target)

            # accumulate to global
            correct_total += correct
            top_5_total += top_5
            test_len_total += max(test_len, 1)

            # for this client
            test_len = max(test_len, 1)
            # loss function averages over batch size
            test_loss /= len(test_data)

            sum_loss = test_loss * test_len

            acc = round(correct / test_len, 4)
            acc_5 = round(top_5 / test_len, 4)
            test_loss = round(test_loss, 4)

            test_loss_total += test_loss

            logging.info('Client #{}: Test set: Average loss: {}, Top-1 Accuracy: {}/{} ({}), Top-5 Accuracy: {}'
                         .format(client_id, test_loss, correct, len(test_data.dataset), acc, acc_5))

            testRes = {'top_1': correct, 'top_5': top_5, 'test_loss': sum_loss, 'test_len': test_len}

        # global
        acc_total = round(correct_total / test_len_total, 4)
        acc_5_total = round(top_5_total / test_len_total, 4)
        test_loss_total = round(test_loss_total, 4)

        logging.info('GLOBAL #{}: Test set: Average loss: {}, Top-1 Accuracy: {}/{} ({}), Top-5 Accuracy: {}'
                     .format(len(test_datas), test_loss_total, correct_total, test_len_total, acc_total, acc_5_total))

        return test_loss, acc, acc_5, testRes


def test_model(rank, model, test_data, device='cpu', criterion=nn.NLLLoss(), tokenizer=None):

    test_loss = 0
    correct = 0
    top_5 = 0

    test_len = 0

    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0

    model = model.to(device=device) # load by pickle
    model.eval()

    with torch.no_grad():

        for data, target in test_data:
            try:
                data, target = Variable(data).to(device=device), Variable(target).to(device=device)

                output = model(data)
                loss = criterion(output, target)

                test_loss += loss.data.item()  # Variable.data
                acc = accuracy(output, target, topk=(1, 5))

                correct += acc[0].item()
                top_5 += acc[1].item()
            
            except Exception as ex:
                logging.info(f"Testing of failed as {ex}")
                break
            test_len += len(target)
    
    test_len = max(test_len, 1)
    # loss function averages over batch size
    # 一个batch的loss，相当于per-sample的loss
    test_loss /= len(test_data)

    sum_loss = test_loss * test_len

    # in NLP, we care about the perplexity of the model
    acc = round(correct / test_len, 4)
    acc_5 = round(top_5 / test_len, 4)
    test_loss = round(test_loss, 4)

    logging.info('Rank {}: Test set: Average loss: {}, Top-1 Accuracy: {}/{} ({}), Top-5 Accuracy: {}'
          .format(rank, test_loss, correct, len(test_data.dataset), acc, acc_5))

    testRes = {'top_1':correct, 'top_5':top_5, 'test_loss':sum_loss, 'test_len':test_len}

    return test_loss, acc, acc_5, testRes

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)

        return res

class RandomParams(object):

    def __init__(self, ratio: float):
        self.ratio = ratio

    def get(self, params_indices: list):
        rng = random.Random()
        rng.seed(random.random() * 1234)
        indexes = [x for x in range(len(params_indices))]
        rng.shuffle(indexes)
        # print(indexes)

        part_len = int(math.floor(self.ratio * len(params_indices)))
        result = indexes[0: part_len]
        return [params_indices[i] for i in result]
