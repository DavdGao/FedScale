# Standard libs
import os, re, shutil, sys, time, datetime, logging, pickle, json, socket
import random, math, gc, copy
from collections import OrderedDict
from multiprocessing import Process, Value
from multiprocessing.managers import BaseManager
import multiprocessing, threading
import numpy as np
import collections
import numpy

# libs from fedscale
from fedscale.core.arg_parser import args
from fedscale.core.utils.utils_data import get_data_transform
from fedscale.core.utils.utils_model import test_model
from fedscale.core.utils.divide_data import select_dataset, DataPartitioner
from fedscale.core.client_manager import clientManager
from fedscale.core.utils.yogi import YoGi
from fedscale.core.optimizer import ServerOptimizer

# PyTorch libs
import torch
from torch.multiprocessing import Process
from torch.multiprocessing import Queue
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as tormodels
from torch.utils.data.sampler import WeightedRandomSampler
from fedscale.core.utils.twitter.leaf import LocalDataset

tokenizer = None
if args.task == 'nlp' or args.task == 'text_clf':
    from fedscale.core.utils.nlp import mask_tokens, load_and_cache_examples
    from transformers import (
        AdamW,
        AutoConfig,
        AlbertTokenizer,
        AutoTokenizer,
        MobileBertForPreTraining,
        AutoModelWithLMHead
    )
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)
elif args.task == 'speech':
    import numba
    from fedscale.core.utils.speech import SPEECH
    from fedscale.core.utils.transforms_wav import ChangeSpeedAndPitchAudio, ChangeAmplitude, FixAudioLength, ToMelSpectrogram, LoadAudio, ToTensor
    from fedscale.core.utils.transforms_stft import ToSTFT, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, FixSTFTDimension, ToMelSpectrogramFromSTFT, DeleteSTFT, AddBackgroundNoiseOnSTFT
    from fedscale.core.utils.speech import BackgroundNoiseDataset
elif args.task == 'detection':
    import pickle
    from fedscale.core.utils.rcnn.lib.roi_data_layer.roidb import combined_roidb
    from fedscale.core.utils.rcnn.lib.datasets.factory import get_imdb
    from fedscale.core.utils.rcnn.lib.datasets.pascal_voc import readClass
    from fedscale.core.utils.rcnn.lib.roi_data_layer.roibatchLoader import roibatchLoader
    from fedscale.core.utils.rcnn.lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
    from fedscale.core.utils.rcnn.lib.model.utils.net_utils import weights_normal_init, save_net, load_net, \
        adjust_learning_rate, save_checkpoint, clip_gradient
    from fedscale.core.utils.rcnn.lib.model.faster_rcnn.resnet import resnet
    from fedscale.core.utils.rcnn.lib.model.rpn.bbox_transform import clip_boxes
    from fedscale.core.utils.rcnn.lib.model.roi_layers import nms
    from fedscale.core.utils.rcnn.lib.model.rpn.bbox_transform import bbox_transform_inv
elif args.task == 'voice':
    from torch_baidu_ctc import CTCLoss
elif args.task == 'rl':
    import gym
    from fedscale.core.utils.dqn import *

# shared functions of aggregator and clients
# initiate for nlp

os.environ['MASTER_ADDR'] = args.ps_ip
os.environ['MASTER_PORT'] = args.ps_port


outputClass = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47,'amazon':5,
                'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5, 'inaturalist' : 1010,
               'twitter': 2
            }

def init_model():
    global tokenizer

    logging.info("Initializing the model ...")

    if args.model == "convnet2":
        from fedscale.core.utils.models import ConvNet2
        logging.info("Load ConvNet2...")
        if args.data_set == "cifar10":
            h, w = 32, 32
        elif args.data_set == "femnist":
            h, w = 28, 28
        model = ConvNet2(args.input_dim, h=h, w=w, hidden=args.hidden, class_num=outputClass[args.data_set],
                         dropout=args.dropout)
    elif args.model == "lr":
        from fedscale.core.utils.models import LogisticRegression
        model = LogisticRegression(input_dim=args.input_dim, output_dim=outputClass[args.data_set])
    else:
        model = tormodels.__dict__[args.model](num_classes=outputClass[args.data_set])

    return model


def init_dataset():

    if args.data_set == 'twitter':
        from fedscale.core.utils.twitter.leaf_twitter import LEAF_TWITTER
        dataset = LEAF_TWITTER(root=args.data_dir,
                               name='twitter',
                               s_frac=0.01,
                               tr_frac=0.8,
                               val_frac=0.1,
                               seed=args.sample_seed)
    else:
        logging.info('DataSet must be {}!'.format(['Mnist', 'Cifar', 'openImg', 'blog', 'stackoverflow', 'speech', 'yelp']))
        sys.exit(-1)

    client_num = len(dataset)

    # get local dataset
    train_data, train_label, test_data, test_label = None, None, None, None
    train_partitions = []
    index = 0
    for client_idx in range(client_num):
        train_client_data = dataset[client_idx]['train'].Xs
        train_client_label = dataset[client_idx]['train'].targets
        if train_data is None:
            train_data = train_client_data
            train_label = train_client_label
        else:
            train_data = np.concatenate([train_data, train_client_data], axis=0)
            train_label = np.concatenate([train_label, train_client_label], axis=0)
        train_partitions += [[_ for _ in range(index, index + len(train_client_data))]]
        index += len(train_client_data)

        if 'test' in dataset[client_idx]:
            test_client_data = dataset[client_idx]['test'].Xs
            test_client_label = dataset[client_idx]['test'].targets

            if test_data is None:
                test_data = test_client_data
                test_label = test_client_label
            else:
                test_data = np.concatenate([test_data, test_client_data], axis=0)
                test_label = np.concatenate([test_label, test_client_label], axis=0)


    training_sets = LocalDataset(Xs=train_data, targets=train_label)
    testing_sets = LocalDataset(Xs=test_data, targets=test_label)

    return training_sets, train_partitions, testing_sets, [[_ for _ in range(len(testing_sets))]]
