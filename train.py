import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse
import utils
import torchvision
from torchvision import datasets, models, transforms
from glob import glob
# import apex.amp as amp
import torch.backends.cudnn as cudnn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader, Dataset

import pandas as pd
from typing import Any, Optional, Tuple
import multiprocessing
import tensorflow as tf

MIN_SAMPLES_PER_CLASS = 25
SELECT_CATEGORIES = [36]
IMAGES_DIR = "/home/daniel/projects/transfer_learning/sample_files"
NUMBER_OF_TRAIN_IMAGES_PER_CLASS = 1000
ORIGINAL_IMAGE_SIZE = 299
IMAGE_SIZE = 299
BATCH_SIZE = 256
NUM_WORKERS = int(multiprocessing.cpu_count() / 2)
BATCH_SIZE = 25
RESNET_SIZE = 18
PREDICT_ONLY = False
LEARNING_RATE = 1e-3
LR_STEP = 3
LR_FACTOR = 0.5
USE_PARALLEL = False
NUM_EPOCHS = 40
LOG_FREQ = 500
NUM_TOP_PREDICTS = 1
MAX_STEPS_PER_EPOCH = 2 ** 32
CHECKPOINT_PATH = "checkpoints"
CHECKPOINT_NAME = None

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
		filename = "COCO_train2014_{}.jpg".format(str(self.df.image_id.values[index]).zfill(12))

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

def load_data(checkpoint: any = None) -> 'Tuple[DataLoader[np.ndarray], DataLoader[np.ndarray], LabelEncoder, int]':
	label_column = "category_id"
	torch.multiprocessing.set_sharing_strategy('file_system')
	cudnn.benchmark = True

	# only use classes which have at least MIN_SAMPLES_PER_CLASS samples
	print('loading data...')
	df = pd.read_csv("sample_images_labels.txt")
	image_files = ["COCO_train2014_{}.jpg".format(str(x).zfill(12)) for x in df["image_id"].tolist()]

	counts = df[label_column].value_counts()
	selected_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
	num_classes = selected_classes.shape[0]
	print('classes with at least N samples:', num_classes)
	train_df = df.loc[df["category_id"].isin(selected_classes)].copy()
	print('train_df', train_df.shape)
	train_exists = lambda img: os.path.exists(f'{IMAGES_DIR}/COCO_train2014_{str(img).zfill(12)}.jpg')
	train_df = train_df.loc[train_df["image_id"].apply(train_exists)].copy()
	print('train_df after filtering', train_df.shape)
	train_df = train_df.loc[df[label_column].isin(SELECT_CATEGORIES)].copy()
	print("Train shape after filtering classes: ", train_df.shape)



	if checkpoint != None:
		print("Loading label encoder from checkpoint...")
		label_encoder = checkpoint["label_encoder"]
	else:
		label_encoder = LabelEncoder()
		label_encoder.fit(train_df.category_id.values)

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

	print(f"Train length: {len(train_df)} Val length: {len(val_df)} Test length: {len(test_df)}")

#     test_df = pd.read_csv(csv_dir + 'test2.csv', dtype=str)
#     print('test_df', test_df.shape)

#     # filter non-existing test images
#     exists = lambda img: os.path.exists(f'/hdd/kaggle/landmarks/test_images2/{img}.jpg')

#     test_df = test_df.loc[test_df.id.apply(exists)].copy()
#     print('test_df after filtering', test_df.shape)
#     assert test_df.shape[0] > 112000
#     # assert test_df.shape[0] > 117703
#     if PREDICT_ONLY:
#         num_classes = len(label_encoder.classes_)
#     print('found classes', len(label_encoder.classes_))
#     assert len(label_encoder.classes_) == num_classes

#     train_df.landmark_id = label_encoder.transform(train_df.landmark_id)
#     val_df.landmark_id = label_encoder.transform(val_df.landmark_id)

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

	return train_loader, val_loader, test_loader, label_encoder, num_classes, dataset_sizes

def accuracy(predicts, targets, confs):

	assert len(predicts.shape) == 1
	assert len(confs.shape) == 1
	assert len(targets.shape) == 1
	assert predicts.shape == confs.shape and confs.shape == targets.shape

	_, indices = torch.sort(confs, descending=True)
	confs = confs.cpu().numpy()
	predicts = predicts[indices].cpu().numpy()
	targets = targets[indices].cpu().numpy()

	num_correct = 0
	for i in range(predictions):
		if predictions[i] == targets[i]:
			num_correct += 1
			
	return num_correct / len(predictions)
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

	running_corrects = 0
	for i, (input_, target) in enumerate(train_loader):
		global_step += 1
		if i >= num_steps:
			break

		output = model(input_.cuda())
		loss = criterion(output, target.cuda())

		confs, predicts = torch.max(output.detach(), dim=1)
		# avg_score.update(accuracy(predicts, target, confs))

		losses.update(loss.data.item(), input_.size(0))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if i % LOG_FREQ == 0:
			tensorboard.log_scalar("train_step_loss", losses.val, global_step)
			# tensorboard.log_scalar("train_step_gap", avg_score.val, global_step)

			print(f'{epoch} [{i}/{num_steps}]\t'
						f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
						# f'GAP {avg_score.val:.4f} ({avg_score.avg:.4f})'
						+ lr_str)

		if has_time_run_out():
			break

	avg_epoch_loss = losses.avg
	# avg_accuracy = avg_score.avg

	print(f"Epoch loss: {avg_epoch_loss}")
#     avg_epoch_gap = avg_score.avg

#     tensorboard.log_scalar("train_epoch_loss", avg_epoch_loss, epoch)
#     tensorboard.log_scalar("train_epoch_gap", avg_epoch_gap, epoch)

#     torch.save({
#         'epoch': epoch,
#         'classifier': model.fc,
#         'model_state_dict': model.state_dict(),
#         'optimizer': optimizer,
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': losses.avg,
#         'gap': avg_score.avg,
#         'global_step': global_step,
#         'label_encoder': label_encoder,
#         'resnet_size': RESNET_SIZE,
#         'image_size': IMAGE_SIZE
#     }, CHECKPOINT_PATH + "checkpoints_{}".format(epoch))

#     print(f' * average GAP on train {avg_score.avg:.4f}')
	

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

if __name__ == "__main__":
	tensorboard = Tensorboard(CHECKPOINT_PATH + "logdir")
	epoch = 1

	if CHECKPOINT_NAME != None:
		checkpoint = torch.load(CHECKPOINT_PATH + CHECKPOINT_NAME)
		train_loader, val_loader, test_loader, label_encoder, num_classes,_ = load_data(checkpoint)
	else:
		train_loader, val_loader, test_loader, label_encoder, num_classes, _ = load_data()

	if RESNET_SIZE == 50:
		model = torchvision.models.resnet50(pretrained=True)
	elif RESNET_SIZE == 101:
		model = torchvision.models.resnet101(pretrained=True)
	elif RESNET_SIZE == 18:
		model = torchvision.models.resnet18(pretrained=True)
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
