
import csv
import os
import numpy as np
import collections

path_base = '/mnt/gaodawei.gdw/FedScale/dataset/data/femnist/client_data_mapping/'

def func(split):
    client_dict = collections.defaultdict(list)
    a = set()
    first = True
    cnt = 0
    for row in csv.reader(open(path_base + '{}.csv'.format(split), 'r'), delimiter=','):
        if first:
            first = False
            continue
        client_id = row[0]
        a.add(client_id)
        client_dict[client_id].append(row[1])
        cnt += 1
    print(split, 'number of clients', len(client_dict), len(a), 'number of rows', cnt)

    for i in range(3597):
        if str(i) not in a:
            print(i, 'not in data')

    return list(client_dict.keys())

def partition():
    path_femnist = path_base + 'femnist.csv'
    path_save = path_base + '{}.csv'

    clients = collections.defaultdict(list)

    header = None
    for row in csv.reader(open(path_femnist), delimiter=','):
        if header is None:
            header = row
            continue
        client_id = int(row[0])

        # femnist中的client_id是从0开始计数的，并且有3597个
        clients[client_id].append(row)

    # 按顺序进行划分和存储
    csv_train = csv.writer(open(path_save.format('train'), 'w'), delimiter=',')
    csv_test = csv.writer(open(path_save.format('test'), 'w'), delimiter=',')
    csv_val = csv.writer(open(path_save.format('val'), 'w'), delimiter=',')
    csv_train.writerow(header)
    csv_test.writerow(header)
    csv_val.writerow(header)
    for client_id in sorted(clients.keys()):
        # shuffle
        np.random.shuffle(clients[client_id])
        # partition
        rows = clients[client_id]
        len_data = len(rows)
        train_split, test_split, val_split = rows[:int(len_data*0.6)], rows[int(len_data*0.6):int(len_data*0.8)], rows[int(len_data*0.8):]
        # save
        csv_train.writerows(train_split)
        csv_test.writerows(test_split)
        csv_val.writerows(val_split)

if __name__ == '__main__':
    func('train')
    func('test')
    func('val')