import argparse
import os
import shutil

from litdata import merge_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path")
parser.add_argument("--num_proc", default=1)


args = parser.parse_args()

dataset_path = args.dataset_path
num_proc = int(args.num_proc)


subdatasets = [f.path for f in os.scandir(dataset_path) if f.is_dir()]

merge_datasets(subdatasets, dataset_path, num_proc, mode="move")

for path in subdatasets:
    shutil.rmtree(path)
