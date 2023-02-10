# %%
import mindspore.dataset.vision.c_transforms as CV

import os

import numpy as np
import mindspore.dataset as ds

from io import StringIO, BytesIO
from PIL import Image, ImageOps, ImageFilter
import random
from torchvision.transforms import Resize
from copy import deepcopy

EXTENSIONS = ['.jpg', '.png']

class MyGaussianBlur(ImageFilter.Filter):

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class MyCoTransform(object):

    def __init__(self, enc, augment, height, if_from_mindrecord=False):
        self.enc = enc
        self.augment = augment
        self.height = height
        self.if_from_mindrecord = if_from_mindrecord
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # self.ratio = 1.3 # 2,3,4阶段使用
        self.ratio = 1.2 # 1阶段使用
        # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def process_one(self, input, target, height):
        if self.augment:

            # GaussianBlur
            input = input.filter(MyGaussianBlur(radius=random.random()))

            if random.random() > 0.5: # random crop

                # # # # # # # # # # # # # # # # # # # # # # # # 
                # ratio = random.random() * (self.ratio - 1) + 1 # 2,3,4阶段使用
                ratio = self.ratio # 1阶段使用
                # # # # # # # # # # # # # # # # # # # # # # # # 

                w = int(2048 / ratio)
                h = int(1024 / ratio)
                
                x = int(random.random()*(2048-w))
                y = int(random.random()*(1024-h))
                
                box = (x, y, x+w, y+h)
                input = input.crop(box)
                target = target.crop(box)

            input =  Resize(height, Image.BILINEAR)(input)
            target = Resize(height, Image.NEAREST)(target)

            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            input =  Resize(height, Image.BILINEAR)(input)
            target = Resize(height, Image.NEAREST)(target)

        input = np.array(input).astype(np.float32) / 255
        input = input.transpose(2, 0, 1)

        if (self.enc):
            target = Resize(int(height/8), Image.NEAREST)(target)
        target = np.array(target).astype(np.uint32)
        target[target == 255] = 19
        return input, target

    def process_one_infer(self, input, height):
        input = Resize(height, Image.BILINEAR)(input)
        input = np.array(input).astype(np.float32) / 255
        input = input.transpose(2, 0, 1)
        return input

    def __call__(self, input, target=None):
        if self.if_from_mindrecord:
            input = Image.open(BytesIO(input))
            target = Image.open(BytesIO(target))
        if target == None:
            input = self.process_one_infer(input, self.height)
            return input
        input, target = self.process_one(input, target, self.height)
        return input, target

class cityscapes:

    def __init__(self, root, subset, enc, aug, height):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')

        self.images_root += subset
        self.labels_root += subset

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.transform = MyCoTransform(enc, aug, height)

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')
        image, label = self.transform(image, label)        
        return image, label

    def __len__(self):
        return len(self.filenames)

class cityscapes_datapath:

    def __init__(self, root, subset):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')

        self.images_root += subset
        self.labels_root += subset

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        return filename, filenameGt

    def __len__(self):
        return len(self.filenames)

class InderDataSet:

    def __init__(self, file_path, height):
        enc = False
        aug = False
        self.transform = MyCoTransform(enc, aug, height)

        with open(file_path, "r") as file:
            lines = file.read()
        lines = lines.split("\n")
        lines = [ line.strip(" ") for line in lines]
        lines = [line for line in lines if line!=""]
        self.data_paths = lines

    def __getitem__(self, index):
        filepath = self.data_paths[index]
        with open(filepath, 'rb') as f:
            image = load_image(f).convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.data_paths)

def getCityScapesDataLoader_GeneratorDataset(CityScapesRoot, subset, batch_size, enc, height, shuffle, aug, rank_id=0, global_size=1, repeat=1):
    dataset = cityscapes(CityScapesRoot, subset, enc, aug, height)
    dataloader = ds.GeneratorDataset(dataset, column_names=["images", "labels"], \
        num_parallel_workers=8, shuffle=shuffle, shard_id=rank_id, num_shards=global_size, python_multiprocessing=True)
    if shuffle:
        dataloader = dataloader.shuffle(batch_size*10)
    dataloader = dataloader.batch(batch_size, drop_remainder=False)
    if repeat > 1:
        dataloader = dataloader.repeat(repeat)
    return dataloader

def getCityScapesDataLoader_mindrecordDataset(data_path, batch_size, enc, height, shuffle, aug, rank_id=0, global_size=1, repeat=1):

    dataloader = ds.MindDataset(data_path, columns_list=["data", "label"], \
        num_parallel_workers=8, shuffle=shuffle, shard_id=rank_id, num_shards=global_size)
    transform = MyCoTransform(enc, aug, height, if_from_mindrecord=True)
    dataloader = dataloader.map(operations=transform, input_columns=["data", "label"], output_columns=["data", "label"],
                   num_parallel_workers=8, python_multiprocessing=True)

    if shuffle:
        dataloader = dataloader.shuffle(batch_size*10)
    dataloader = dataloader.batch(batch_size, drop_remainder=False)
    if repeat > 1:
        dataloader = dataloader.repeat(repeat)
    return dataloader

def getInferDataLoader_fromfile(file_path, batch_size, height):
    shuffle = False
    global_size = 1
    rank_id = 0
    dataset = InderDataSet(file_path, height)
    dataloader = ds.GeneratorDataset(dataset, column_names=["images"], \
                                     num_parallel_workers=8, shuffle=shuffle, shard_id=rank_id, num_shards=global_size,
                                     python_multiprocessing=True)
    dataloader = dataloader.batch(batch_size, drop_remainder=False)
    return dataloader

# %%
if __name__ == "__main__":
    # dataset, dataloader = getCityScapesDataLoader("train", 8, False, \
    #     [416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592], False, False)

    # dataset, dataloader = getCityScapesDataLoader("train", 6, True, \
    #     512, False, False)

    dataset, dataloader = getCityScapesDataLoader("val", 6, False, \
        512, False, True)
    import cv2
    from show import Colorize_cityscapes
    colorize = Colorize_cityscapes()
    for i in dataloader:
        img = i[0] * 255
        img = img.asnumpy().astype(np.uint8)
        img = img[0]
        img = img.transpose(1, 2, 0)
        img = img[:, :, ::-1]
        cv2.imwrite("img.jpg", img)

        mask = i[1][0].asnumpy().astype(np.uint8)
        colorized_label = colorize(mask)
        cv2.imwrite("label.jpg", colorized_label)
        break
