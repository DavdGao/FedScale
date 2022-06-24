# -*- coding: utf-8 -*-
from random import Random
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
import random, csv
from collections import defaultdict
#from argParser import args


def _split_according_to_prior(label, client_num, prior):
    assert client_num == len(prior)
    classes = len(np.unique(label))
    assert classes == len(np.unique(np.concatenate(prior, 0)))

    # counting
    frequency = np.zeros(shape=(client_num, classes))
    for idx, client_prior in enumerate(prior):
        for each in client_prior:
            frequency[idx][each] += 1
    sum_frequency = np.sum(frequency, axis=0)

    idx_slice = [[] for _ in range(client_num)]
    for k in range(classes):
        idx_k = np.where(label == k)[0]
        np.random.shuffle(idx_k)
        nums_k = np.ceil(frequency[:, k]/sum_frequency[k]*len(idx_k)).astype(int)
        while len(idx_k) < np.sum(nums_k):
            random_client = np.random.choice(range(client_num))
            if nums_k[random_client] > 0:
                nums_k[random_client]-=1
        assert len(idx_k) == np.sum(nums_k)
        idx_slice = [idx_j + idx.tolist() for idx_j, idx in zip(idx_slice, np.split(idx_k, np.cumsum(nums_k)[:-1]))]

    for i in range(len(idx_slice)):
        np.random.shuffle(idx_slice[i])
    return idx_slice

def dirichlet_distribution_noniid_slice(label, client_num, alpha, min_size=10, prior=None):
    r"""Get sample index list for each client from the Dirichlet distribution.
    https://github.com/FedML-AI/FedML/blob/master/fedml_core/non_iid_partition/noniid_partition.py
    Arguments:
        label (np.array): Label list to be split.
        client_num (int): Split label into client_num parts.
        alpha (float): alpha of LDA.
        min_size (int): min number of sample in each client
    Returns:
        idx_slice (List): List of splited label index slice.
    """
    if len(label.shape) != 1:
        raise ValueError('Only support single-label tasks!')

    if prior is not None:
        return _split_according_to_prior(label, client_num, prior)

    num = len(label)
    classes = len(np.unique(label))
    assert num > client_num * min_size, f'The number of sample should be greater than {client_num * min_size}.'
    size = 0
    while size < min_size:
        idx_slice = [[] for _ in range(client_num)]
        for k in range(classes):
            # for label k
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            prop = np.random.dirichlet(np.repeat(alpha, client_num))
            #prop = np.array([
            #    p * (len(idx_j) < num / client_num)
            #    for p, idx_j in zip(prop, idx_slice)
            #])
            #prop = prop / sum(prop)
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
            idx_slice = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))
            ]
            size = min([len(idx_j) for idx_j in idx_slice])
    for i in range(client_num):
        np.random.shuffle(idx_slice[i])
    return idx_slice


class Partition(object):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partition data by trace or random"""

    def __init__(self, data, args, numOfClass=0, seed=10, isTest=False):
        self.partitions = []
        self.partitions_test = []
        self.rng = Random()
        self.rng.seed(seed)

        self.data = data
        self.labels = self.data.targets
        self.args = args
        self.isTest = isTest
        np.random.seed(seed)

        self.data_len = len(self.data)
        self.task = args.task
        self.numOfLabels = numOfClass
        self.client_label_cnt = defaultdict(set)

    def getNumOfLabels(self):
        return self.numOfLabels

    def getDataLen(self):
        return self.data_len

    def getClientLen(self):
        return len(self.partitions)

    def getClientLabel(self):
        return [len(self.client_label_cnt[i]) for i in range( self.getClientLen())]

    def trace_partition(self, data_map_file):
        """Read data mapping from data_map_file. Format: <client_id, sample_name, sample_category, category_id>"""
        logging.info(f"Partitioning data by profile {data_map_file}...")

        clientId_maps = {}
        unique_clientIds = {}
        # load meta data from the data_map_file
        with open(data_map_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            read_first = True
            sample_id = 0

            for row in csv_reader:
                if read_first:
                    logging.info(f'Trace names are {", ".join(row)}')
                    read_first = False
                else:
                    client_id = row[0]

                    if client_id not in unique_clientIds:
                        unique_clientIds[client_id] = len(unique_clientIds)

                    clientId_maps[sample_id] = unique_clientIds[client_id]
                    self.client_label_cnt[unique_clientIds[client_id]].add(row[-1])
                    sample_id += 1

        # Partition data given mapping
        self.partitions = [[] for _ in range(len(unique_clientIds))]

        for idx in range(sample_id):
            self.partitions[clientId_maps[idx]].append(idx)

    def dirichlet_partition(self, client_num, alpha, min_size=1, prior=None, isTest=False):
        distribution = dirichlet_distribution_noniid_slice(
            label=np.array(self.labels),
            client_num=client_num,
            alpha=alpha,
            min_size=min_size,
            prior=prior
        )

        if isTest:
            self.partitions_test = distribution
            self.uniform_partition(num_clients=1)
        else:
            self.partitions = distribution

            # deal with the prior distribution
            client_labels = list()
            for idx in distribution:
                labels = [self.labels[id] for id in idx]
                client_labels.append(labels)

            return client_labels

    def uniform_partition(self, num_clients):
        # random partition
        numOfLabels = self.getNumOfLabels()
        data_len = self.getDataLen()
        logging.info(f"Randomly partitioning data, {data_len} samples...")

        indexes = list(range(data_len))
        self.rng.shuffle(indexes)

        for _ in range(num_clients):
            part_len = int(1./num_clients * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition, istest):
        resultIndex = self.partitions[partition]

        exeuteLength = len(resultIndex) if not istest else int(len(resultIndex) * self.args.test_ratio)
        resultIndex = resultIndex[:exeuteLength]
        self.rng.shuffle(resultIndex)

        return Partition(self.data, resultIndex)


    def getSize(self):
        # return the size of samples
        return {'size': [len(partition) for partition in self.partitions]}


def select_dataset(rank, partition, batch_size, args, isTest=False, collate_fn=None):
    """Load data given client Id"""
    partition = partition.use(rank - 1, isTest)
    dropLast = False
    num_loaders = min(int(len(partition)/args.batch_size/2), args.num_loaders)
    if num_loaders == 0:
        time_out = 0
    else:
        time_out = 60

    if collate_fn is not None:
        return DataLoader(partition, batch_size=batch_size, shuffle=not isTest, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast, collate_fn=collate_fn)
    return DataLoader(partition, batch_size=batch_size, shuffle=not isTest, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast)


