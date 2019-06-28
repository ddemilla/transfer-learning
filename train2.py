#!/usr/bin/env python
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]=["0","1"]
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import multiprocessing
import time

import numpy as np
import numpy
import pandas as pd

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm

import io
from PIL import Image
import tensorflow as tf

from sklearn.model_selection import train_test_split
from random import random
from skimage.io import imread
from PIL import ImageDraw 
import apex.amp as amp

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
MIN_SAMPLES_PER_CLASS = 50
BATCH_SIZE = 256
# BATCH_SIZE = 100
# BATCH_SIZE = 32
LEARNING_RATE = 1e-3
LR_STEP = 3
LR_FACTOR = 0.5
NUM_WORKERS = int(multiprocessing.cpu_count() / 2)
# MAX_STEPS_PER_EPOCH = 15000
MAX_STEPS_PER_EPOCH = 2 ** 32
# NUM_EPOCHS = 2 ** 32
NUM_EPOCHS = 40
LOG_FREQ = 500
NUM_TOP_PREDICTS = 1
# TIME_LIMIT = 9 * 60 * 60
TIME_LIMIT = 500 * 60 * 60
IMAGE_SIZE = 64
ORIGINAL_IMAGE_SIZE = 299
RESNET_SIZE = 50
USE_PARALLEL = False
IMAGES_DIR = "/home/daniel/projects/transfer_learning/sample_files"
SELECT_CATEGORIES = [36]
NUMBER_OF_TRAIN_IMAGES_PER_CLASS = 1000
# CHECKPOINT_PATH = f"checkpoints/checkpoints_{IMAGE_SIZE}_{RESNET_SIZE}_cleaned_fp16/"
# CHECKPOINT_PATH = f"checkpoints/checkpoints_{ORIGINAL_IMAGE_SIZE}_{IMAGE_SIZE}_{RESNET_SIZE}_{MIN_SAMPLES_PER_CLASS}_{NUM_EPOCHS}/"
CHECKPOINT_PATH = f"checkpoints/"
print(CHECKPOINT_PATH)
# CHECKPOINT_PATH = f"checkpoints/checkpoints_128_50/"

if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

# CHECKPOINT_NAME = None
CHECKPOINT_NAME = None

# PREDICTION_FILE = f"submission_{IMAGE_SIZE}_{RESNET_SIZE}_cleaned_fp16/"
PREDICTION_FILE = f"submission_{ORIGINAL_IMAGE_SIZE}_{IMAGE_SIZE}_{RESNET_SIZE}_{MIN_SAMPLES_PER_CLASS}_{NUM_EPOCHS}"
TRAIN_CSV = "train_cleaned"
# TRAIN_CSV = "train"

csv_dir = "/home/daniel/kaggle/landmarks/csv/"
PREDICT_ONLY = False
# EXPANDED_DESCRIPTIONS = pd.read_csv("csv/expanded_landmarks_descriptions.csv", names=["id", "description"])

class Tensorboard:
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def close(self):
        self.writer.close()

    def log_scalar(self, tag, value, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()
        
    def log_histogram(self, tag, values, global_step, bins):
        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary()
        summary.value.add(tag=tag, histo=hist)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_image(self, tag, img, global_step):
        s = io.BytesIO()
        Image.fromarray(img).save(s, format='png')

        img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=img.shape[0],
                                   width=img.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_plot(self, tag, figure, global_step):
        plot_buf = io.BytesIO()
        figure.savefig(plot_buf, format='png')
        plot_buf.seek(0)
        img = Image.open(plot_buf)
        img_ar = np.array(img)

        img_summary = tf.Summary.Image(encoded_image_string=plot_buf.getvalue(),
                                   height=img_ar.shape[0],
                                   width=img_ar.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, mode: str) -> None:
        print(f'creating data loader - {mode}')
        assert mode in ['train', 'val', 'test']

        self.df = dataframe
        self.mode = mode

        if self.mode == "test":
            transforms_list = [
                transforms.CenterCrop(ORIGINAL_IMAGE_SIZE),
                transforms.Resize(IMAGE_SIZE)
            ]
        elif self.mode == "train":
            transforms_list = [
                # transforms.Resize(IMAGE_SIZE)
                transforms.Resize(ORIGINAL_IMAGE_SIZE)
            ]
        else:
            transforms_list = [
                transforms.Resize(IMAGE_SIZE)
                # transforms.Resize(ORIGINAL_IMAGE_SIZE)
            ]

        if self.mode == 'train':
            transforms_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(IMAGE_SIZE),
                transforms.RandomChoice([
                    transforms.RandomChoice([
                        transforms.RandomResizedCrop(IMAGE_SIZE),
                        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                        transforms.RandomAffine(degrees=15, translate=(0.2, 0.2),
                                                scale=(0.8, 1.2), shear=15,
                                                resample=Image.BILINEAR)
                    ]),
                    transforms.RandomChoice([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.RandomRotation(degrees=90),
                        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                        transforms.RandomAffine(degrees=15, translate=(0.2, 0.2),
                                                scale=(0.8, 1.2), shear=15,
                                                resample=Image.BILINEAR)
                    ])
                ])
            ])


        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
        self.transforms = transforms.Compose(transforms_list)

    def __getitem__(self, index: int) -> Any:
        ''' Returns: tuple (sample, target) '''
#         filename = self.df.image_id.values[index]
        filename = "COCO_train2014_{}.jpg".format(str(self.df["image_id"].values[index]).zfill(12))

        part = 1 if self.mode == 'test' or filename[0] in '01234567' else 2
#         directory = 'test' if self.mode == 'test' else 'train_' + filename[0]
        # sample = Image.open(f'../input/google-landmarks-2019-64x64-part{part}/{directory}/{self.mode}_64/{filename}.jpg')
        if self.mode == "train":
            sample = Image.open(f'{IMAGES_DIR}/{filename}')
        elif self.mode == "val":
            sample = Image.open(f'{IMAGES_DIR}/{filename}')
        else:
            sample = Image.open(f'{IMAGES_DIR}/{filename}')
        assert sample.mode == 'RGB'

        image = self.transforms(sample)

        if self.mode == 'test':
            return image
        else:
            return image, self.df["category_id"].values[index]

    def __len__(self) -> int:
        return self.df.shape[0]

def GAP(predicts: torch.Tensor, confs: torch.Tensor, targets: torch.Tensor) -> float:
    ''' Simplified GAP@1 metric: only one prediction per sample is supported '''
    assert len(predicts.shape) == 1
    assert len(confs.shape) == 1
    assert len(targets.shape) == 1
    assert predicts.shape == confs.shape and confs.shape == targets.shape

    _, indices = torch.sort(confs, descending=True)

    confs = confs.cpu().numpy()
    predicts = predicts[indices].cpu().numpy()
    targets = targets[indices].cpu().numpy()

    res, true_pos = 0.0, 0

    for i, (c, p, t) in enumerate(zip(confs, predicts, targets)):
        rel = int(p == t)
        true_pos += rel

        res += true_pos / (i + 1) * rel

    res /= targets.shape[0] # FIXME: incorrect, not all test images depict landmarks
    return res

class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_data(checkpoint: any = None) -> 'Tuple[DataLoader[np.ndarray], DataLoader[np.ndarray], LabelEncoder, int]':
    label_column = "category_id"
    torch.multiprocessing.set_sharing_strategy('file_system')
    cudnn.benchmark = True

    # only use classes which have at least MIN_SAMPLES_PER_CLASS samples
    print('loading data...')
    df = pd.read_csv("sample_images_labels.txt")
    # image_files = ["COCO_train2014_{}.jpg".format(str(x).zfill(12)) for x in df["image_id"].tolist()]

    counts = df[label_column].value_counts()
    selected_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    num_classes = selected_classes.shape[0]
    print('classes with at least N samples:', num_classes)
    train_df = df.loc[df[label_column].isin(selected_classes)].copy()
    print('train_df', train_df.shape)
    train_exists = lambda img: os.path.exists(f'{IMAGES_DIR}/COCO_train2014_{str(img).zfill(12)}.jpg')
    train_df = train_df.loc[train_df["image_id"].apply(train_exists)].copy()
    print('train_df after filtering', train_df.shape)
    # train_df = train_df.loc[df[label_column].isin(SELECT_CATEGORIES)].copy()
    # print("Train shape after filtering classes: ", train_df.shape)
    # new_counts = train_df[label_column].value_counts()
    # num_classes = new_counts.shape[0]
    print("Final number of classes: ", num_classes)


    if checkpoint != None:
        print("Loading label encoder from checkpoint...")
        label_encoder = checkpoint["label_encoder"]
    else:
        label_encoder = LabelEncoder()
        label_encoder.fit(train_df[label_column].values)

    y = train_df.pop(label_column)
    x = train_df

    train_size = NUMBER_OF_TRAIN_IMAGES_PER_CLASS * y.nunique()
    train_x, val_x, train_y, val_y = train_test_split(x, y, train_size=train_size, random_state=42, stratify=y)
    train_x[label_column] = train_y
    val_x[label_column] = val_y

    train_df = train_x
    val_df = val_x
    
    y = val_df.pop(label_column)
    x = val_df
    val_x, test_x, val_y, test_y = train_test_split(x, y, test_size=0.5, random_state=42, stratify=y)
    val_x[label_column] = val_y
    test_x[label_column] = test_y

    val_df = val_x
    test_df = test_x

    if PREDICT_ONLY:
        num_classes = len(label_encoder.classes_)
    print('found classes', len(label_encoder.classes_))
    assert len(label_encoder.classes_) == num_classes

    train_df[label_column] = label_encoder.transform(train_df[label_column])
    val_df[label_column] = label_encoder.transform(val_df[label_column])
    test_df[label_column] = label_encoder.transform(test_df[label_column])

    print(f"Train length: {len(train_df)} Val length: {len(val_df)} Test length: {len(test_df)}")

    train_dataset = ImageDataset(train_df, mode='train')
    test_dataset = ImageDataset(test_df, mode='test')
    val_dataset = ImageDataset(val_df, mode='val')
    
    dataset_sizes = [len(train_dataset), len(test_dataset), len(val_dataset)]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

    return train_loader, val_loader, test_loader, label_encoder, num_classes

def train(train_loader: Any, model: Any, criterion: Any, optimizer: Any,
          epoch: int, lr_scheduler: Any, tensorboard: Any, label_encoder:Any) -> None:
    print(f'epoch {epoch}')
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    num_steps = min(len(train_loader), MAX_STEPS_PER_EPOCH)

    print(f'total batches: {num_steps}')

    end = time.time()
    lr_str = ''

    global_step = (epoch - 1) * len(train_loader)

    for i, (input_, target) in enumerate(train_loader):
        print(input_.shape)
        global_step += 1
        if i >= num_steps:
            break

        output = model(input_.cuda())
        loss = criterion(output, target.cuda())
        print(output.detach().cpu().numpy())

        confs, predicts = torch.max(output.detach(), dim=1)
        avg_score.update(GAP(predicts, confs, target))

        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % LOG_FREQ == 0:
            tensorboard.log_scalar("train_step_loss", losses.val, global_step)
            tensorboard.log_scalar("train_step_gap", avg_score.val, global_step)

            print(f'{epoch} [{i}/{num_steps}]\t'
                        f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'GAP {avg_score.val:.4f} ({avg_score.avg:.4f})'
                        + lr_str)

        if has_time_run_out():
            break

    avg_epoch_loss = losses.avg
    avg_epoch_gap = avg_score.avg

    tensorboard.log_scalar("train_epoch_loss", avg_epoch_loss, epoch)
    tensorboard.log_scalar("train_epoch_gap", avg_epoch_gap, epoch)

    torch.save({
        'epoch': epoch,
        'classifier': model.fc,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses.avg,
        'gap': avg_score.avg,
        'global_step': global_step,
        'label_encoder': label_encoder,
        'resnet_size': RESNET_SIZE,
        'image_size': IMAGE_SIZE
    }, CHECKPOINT_PATH + "checkpoints_{}".format(epoch))

    print(f' * average GAP on train {avg_score.avg:.4f}')

def inference(data_loader: Any, model: Any) -> Tuple[torch.Tensor, torch.Tensor,
                                                     Optional[torch.Tensor]]:
    ''' Returns predictions and targets, if any. '''
    model.eval()

    activation = nn.Softmax(dim=1)
    all_predicts, all_confs, all_targets, all_predicts_gap, all_confs_gap = [], [], [], [], []

    print("Data loader length", len(data_loader))
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, disable=IN_KERNEL)):
            if data_loader.dataset.mode != 'test':
                input_, target = data
            else:
                input_, target = data, None

            output = model(input_.cuda())
            output = activation(output)

            confs, predicts = torch.topk(output, NUM_TOP_PREDICTS)
            confs_gap, predicts_gap = torch.max(output.detach(), 1)
            all_confs.append(confs)
            all_predicts.append(predicts)
            all_predicts_gap.append(predicts_gap)
            all_confs_gap.append(confs_gap)

            if target is not None:
                all_targets.append(target)

    predicts = torch.cat(all_predicts)
    confs = torch.cat(all_confs)
    targets = torch.cat(all_targets) if len(all_targets) else None
    predicts_gap = torch.cat(all_predicts_gap)
    confs_gap = torch.cat(all_confs_gap)

    return predicts, confs, targets, predicts_gap, confs_gap

# def image_append_text(image_path, landmark_id):
#     landmark_id_decoded = label_encoder.inverse_transform([landmark_id])[0]
#     description = EXPANDED_DESCRIPTIONS[EXPANDED_DESCRIPTIONS["id"] == landmark_id_decoded]["description"].tolist()[0]
#     description = "\n".join(description.split(",")[1:])
#     description = f'{landmark_id_decoded}\n{description}'
#     img = Image.open(image_path)
#     draw = ImageDraw.Draw(img)
#     try:
#         draw.text((0, 0),description.encode("utf-8"),(255,0,255))
#     except:
#         draw.text((0, 0),str(landmark_id_decoded),(255,0,255))
#     return img

# def eval(val_loader: Any, train_loader: Any, model: Any, tensorboard: Any, epoch: int) -> np.ndarray:
#     predicts_gpu, confs_gpu, targets_gpu, predicts_gap_gpu, confs_gap_gpu = inference(val_loader, model)
#     val_gap = GAP(predicts_gap_gpu, confs_gap_gpu, targets_gpu)
#     num_correct = torch.sum(predicts_gap_gpu.cpu() == targets_gpu.cpu())
#     predicts, confs, targets = predicts_gpu.cpu().numpy(), confs_gpu.cpu().numpy(), targets_gpu.cpu().numpy()


#     labels = [label_encoder.inverse_transform(pred) for pred in predicts]

#     assert len(labels) == len(val_loader.dataset.df)

#     print(f"Val GAP: {val_gap}, Num correct: {num_correct}")

#     tensorboard.log_scalar("val_num_correct", num_correct, epoch)
#     tensorboard.log_scalar("val_gap", val_gap, epoch)

#     val_df = val_loader.dataset.df
#     train_df = train_loader.dataset.df
#     rand_idx = int(random() * len(val_df))

#     sample_row = val_df.iloc[rand_idx]
#     sample_target = sample_row["landmark_id"]
#     sample_prediction = int(predicts[rand_idx])

#     sample_predict_image_name = train_df[train_df["landmark_id"] == sample_prediction]["id"].tolist()[0]
#     sample_predict_image_path = f"/home/daniel/kaggle/landmarks/all_images_resized_{ORIGINAL_IMAGE_SIZE}/{sample_predict_image_name}.jpg"
#     sample_correct_label_image_path = f"/home/daniel/kaggle/landmarks/all_images_resized_{ORIGINAL_IMAGE_SIZE}/{sample_row['id']}.jpg"

#     # images = [image_append_text(sample_predict_image_path,sample_prediction), image_append_text(sample_correct_label_image_path,sample_target)]

#     widths, heights = zip(*(i.size for i in images))

#     total_width = sum(widths)
#     max_height = max(heights)

#     new_im = Image.new('RGB', (total_width, max_height))

#     x_offset = 0
#     for im in images:
#         new_im.paste(im, (x_offset,0))
#         x_offset += im.size[0]

#     tensorboard.log_image("predicted_and_target_image", np.asarray(new_im), epoch)

def generate_submission(test_loader: Any, model: Any, label_encoder: Any) -> np.ndarray:
    print("generating a submission")
    sample_sub = pd.read_csv(csv_dir + 'recognition_sample_submission_stage2.csv')

    predicts_gpu, confs_gpu, _, _, _ = inference(test_loader, model)
    predicts, confs = predicts_gpu.cpu().numpy(), confs_gpu.cpu().numpy()

    labels = [label_encoder.inverse_transform([pred]) for pred in predicts]
    print('labels')
    print(np.array(labels))
    print('confs')
    print(np.array(confs))

    sub = test_loader.dataset.df
    def concat(label: np.ndarray, conf: np.ndarray) -> str:
        return ' '.join([f'{L} {c}' for L, c in zip(label, conf)])
    sub['landmarks'] = [concat(label, conf) for label, conf in zip(labels, confs)]
    print(sub[:10])

    sample_sub = sample_sub.set_index('id')
    sub = sub.set_index('id')
    sample_sub.update(sub)

    sample_sub.to_csv(f'predictions/{PREDICTION_FILE}.csv')

def has_time_run_out() -> bool:
    return time.time() - global_start_time > TIME_LIMIT - 500

if __name__ == '__main__':
    tensorboard = Tensorboard(CHECKPOINT_PATH + "logdir")
    epoch = 1

    if CHECKPOINT_NAME != None:
        checkpoint = torch.load(CHECKPOINT_PATH + CHECKPOINT_NAME)
        train_loader, val_loader, test_loader, label_encoder, num_classes = load_data(checkpoint)
    else:
        train_loader, val_loader, test_loader, label_encoder, num_classes = load_data()

    if RESNET_SIZE == 50:
        model = torchvision.models.resnet50(pretrained=True)
    elif RESNET_SIZE == 101:
        model = torchvision.models.resnet101(pretrained=True)
    else:
        raise ValueError("Invalid resnet size: ", RESNET_SIZE)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.cuda()

    if CHECKPOINT_NAME != None:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])

    global_start_time = time.time()



    if not PREDICT_ONLY:
        print("Training...")
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        if CHECKPOINT_NAME != None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP,
                                                       gamma=LR_FACTOR)

        if USE_PARALLEL:
            print("[Using all the available GPUs]")
            model = nn.DataParallel(model, device_ids=[0, 1])

        for epoch in range(epoch, NUM_EPOCHS + 1):
            print('-' * 50)
            train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, tensorboard, label_encoder)
            eval(val_loader, train_loader, model, tensorboard, epoch)
            lr_scheduler.step()

            if has_time_run_out():
                break

        print('inference mode')
    generate_submission(test_loader, model, label_encoder)
