import os
from argparse import ArgumentParser
import numpy as np
from mindspore.mindrecord import FileWriter
from dataset import cityscapes_datapath
from tqdm import tqdm

# python build_mrdata.py --dataset_path /home/gpf/cityscapes/  --subset train --output_name train.mindrecord
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--subset', type=str)
    parser.add_argument('--output_name', type=str)
    config = parser.parse_args()

    output_name = config.output_name
    subset = config.subset
    dataset_path = config.dataset_path
    assert subset == "val" or subset == "train"

    dataPathLoader = cityscapes_datapath(dataset_path, subset)

    writer = FileWriter(file_name=output_name)
    seg_schema = {"file_name": {"type": "string"}, "label": {"type": "bytes"}, "data": {"type": "bytes"}}
    writer.add_schema(seg_schema, "seg_schema")

    data_list = []
    cnt = 0
    for img_path, label_path in tqdm(dataPathLoader):

        sample_ = {"file_name": img_path.split('/')[-1]}

        with open(img_path, 'rb') as f:
            sample_['data'] = f.read()
        with open( label_path, 'rb') as f:
            sample_['label'] = f.read()
        data_list.append(sample_)
        cnt += 1
        if cnt % 100 == 0:
            writer.write_raw_data(data_list)
            print('number of samples written:', cnt)
            data_list = []

    if data_list:
        writer.write_raw_data(data_list)
    writer.commit()
    print('number of samples written:', cnt)
