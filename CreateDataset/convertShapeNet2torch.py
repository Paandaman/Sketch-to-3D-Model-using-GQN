import os
import collections
import torch
import torch.utils.data as data
import gzip
from gqnshapenet import *
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Convert and save data to PyTorch tensor format')
parser.add_argument('--SKETCH_TRAIN_DATASET_PATH', type=str, help='Path to sketches in training dataset', default='.')
parser.add_argument('--SKETCH_TEST_DATASET_PATH', type=str, help='Path to sketches in test dataset', default='.')
parser.add_argument('--GROUND_TRUTH_TRAIN_DATASET_PATH', type=str, help='Path to sketches in test dataset', default='.')
parser.add_argument('--GROUND_TRUTH_TEST_DATASET_PATH', type=str, help='Path to sketches in test dataset', default='.')
parser.add_argument('--TRAIN_OUTPUT_DIR', type=str, help='Path to output directory', default='.')
parser.add_argument('--TEST_OUTPUT_DIR', type=str, help='Path to output directory', default='.')
opt = parser.parse_args()

def arrange_data(data_tmp, M, gen_rand_indx, rand_idx):
    x_tmp = []
    v_tmp = []
    for data in data_tmp:
        x_tmp.append(torch.stack([x[0][0] for x in data]))
        v_tmp.append(torch.stack([v[1][0] for v in data]))
    if gen_rand_indx:
        rand_idx = torch.LongTensor(1).random_(0, 15)

    x_tmp = torch.stack(x_tmp)
    v_tmp = torch.stack(v_tmp)

    v_tmp_xyz = v_tmp.narrow(dim=2, start=0, length=3)
    v_tmp_jawpitch = v_tmp.narrow(dim=2, start=3, length=2)
    v_tmp_classID = v_tmp.narrow(dim=2, start=5, length=1)

    v_tmp_jawpitch_cosed = torch.cos(v_tmp_jawpitch)
    v_tmp_jawpitch_sined = torch.sin(v_tmp_jawpitch)

    v_pitch_cosed = v_tmp_jawpitch_cosed.narrow(dim=2, start=0, length=1)
    v_pitch_sined = v_tmp_jawpitch_sined.narrow(dim=2, start=0, length=1)
    v_jaw_cosed = v_tmp_jawpitch_cosed.narrow(dim=2, start=0, length=1)
    v_jaw_sined = v_tmp_jawpitch_sined.narrow(dim=2, start=0, length=1)

    v_tmp = torch.cat([torch.cat([v_tmp_xyz, v_jaw_cosed], dim=2), v_jaw_sined], dim=2)
    v_tmp = torch.cat([torch.cat([v_tmp, v_pitch_cosed], dim=2), v_pitch_sined], dim=2)
    v_tmp = torch.cat([v_tmp, v_tmp_classID], dim=2)
    
    q_tmp = v_tmp[:,rand_idx, :]

    v_real_query = q_tmp
    critic_img = x_tmp[:,rand_idx, :, :, :]
    x_real = x_tmp.narrow(dim= 1, start=0, length=M)
    v_real = v_tmp.narrow(dim= 1, start=0, length=M)
    
    critic_img = critic_img.squeeze(dim=1)

    return x_real, v_real, v_real_query, rand_idx, critic_img

if __name__ == '__main__':
    import sys

    S_max = 2000000

    path_sketch_train = opt.SKETCH_TRAIN_DATASET_PATH
    path_real_train = opt.GROUND_TRUTH_TRAIN_DATASET_PATH
    path_sketch_test = opt.SKETCH_TEST_DATASET_PATH
    path_real_test = opt.GROUND_TRUTH_TEST_DATASET_PATH

    train_data = Shapenet_dataset(path_sketch_train, path_real_train, batch_size=1)
    # Data loader. Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.
    train_loader = data.DataLoader(train_data, batch_size = 1, shuffle=False, drop_last = True)

    test_data = Shapenet_dataset(path_sketch_test, path_real_test, batch_size=1)
    # Data loader. Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.
    test_loader = data.DataLoader(test_data, batch_size = 1, shuffle=False, drop_last = True)

    train_iter = iter(train_loader)
    test_iter = iter(test_loader)

    save_train = opt.TRAIN_OUTPUT_DIR
    save_test = opt.TEST_OUTPUT_DIR
    tot = 0
    w = 0

    for t in tqdm(range(S_max)):
        try:
            data = next(train_iter)
        except StopIteration:
            print("DONE!")
            break
        save_path = save_train + str(w) + '.pt.gz'
        w += 1
        with gzip.open(save_path, 'wb') as f:
            torch.save(data, f)

    print(f' [-] {w} scenes in the train dataset')

    ## test

    d = 0
    for t in tqdm(range(S_max)):
        try:
            data = next(test_iter)
        except StopIteration:
            print("DONE!")
            break
        save_path = save_test + str(d) + '.pt.gz'
        d += 1
        with gzip.open(save_path, 'wb') as f:
            torch.save(data, f)
    print(f' [-] {d} scenes in the test dataset')
