import argparse
import glob
import os
import random
import json

def parse_args():
    """
    example
    :param data_root: /root/SSD/PiWiFi/NYCU
    :param person_number: 1
    :param mode: train (or val, test. train -> train&val.json, val -> val.json, test -> test.json)
    """
    parser = argparse.ArgumentParser(description="Split data script")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the data')
    parser.add_argument('--person_number', type=int, required=True, help='Number of people')
    parser.add_argument('--mode', type=str, default='train', help='Dataset mode')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Data root directory: {args.data_root}")
    print(f"Number of people: {args.person_number}")
    print(f"Dataset mode: {args.mode}")

    if args.person_number == 1:
        codes = ['F1', 'F2', 'F3', 'M1', 'M2', 'M3', 'N']

    # get the json file name
    if args.mode == 'train':
        json_name = 'train&val.json'
    elif args.mode == 'val':
        json_name = 'val.json'
    elif args.mode == 'test':
        json_name = 'test.json'

    # get all file path
    all_npy_paths = []
    for c in codes:
        if args.mode == 'train':
            c_np_paths = glob.glob(os.path.join(args.data_root, 'Env*', 'npy', c, '*', '*', '*.npz'))
        elif args.mode == 'val':
            c_np_paths = glob.glob(os.path.join(args.data_root, 'val_set', 'npy', c, '*', '*', '*.npz'))
        elif args.mode == 'test':
            c_np_paths = glob.glob(os.path.join(args.data_root, 'test_set', 'npy', c, '*', '*', '*.npz'))
        all_npy_paths += c_np_paths
    print(f"Total number of npz files: {len(all_npy_paths)}")

    # shuffle and split data, if mode is train, split into train and val
    if args.mode == 'train':
        random.shuffle(all_npy_paths)
        split_ratio = 0.8
        split_idx = int(len(all_npy_paths) * split_ratio)
        datas = {}
        datas['train'] = all_npy_paths[:split_idx]
        datas['val'] = all_npy_paths[split_idx:]
        print(f"Number of training data: {len(datas['train'])}")
        print(f"Number of validation data: {len(datas['val'])}")
    else:
        datas = {}
        datas[args.mode] = all_npy_paths
        print(f"Number of {args.mode} data: {len(datas[args.mode])}")

    # write to a json file
    with open(json_name, 'w') as f:
        json.dump(datas, f)

if __name__ == "__main__":
    main()