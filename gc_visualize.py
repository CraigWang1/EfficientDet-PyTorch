import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
import efficientdet
import matplotlib
import matplotlibt.pyplot as plt

import sys
import cv2
import google.colab.patches as gc   #make showing images compatible with google colab (gc)
#from google.colab.patches import cv2_imshow()

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer


#assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--scaling-compound', help='EfficientDet scaling compound phi.', type=int, default=0)
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--model', help='Path to model (.pt) file.')

	parser = parser.parse_args(args)

	img_size = parser.scaling_compound * 128 + 512
	
	if parser.dataset == 'coco':  #change this later, match it up with train.py
		dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer(img_size=img_size)]))
	elif parser.dataset == 'csv':
		dataset_val = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer(img_size=img_size)]))
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
	
	#load in the model
	device = torch.device('cpu')
	retinanet = efficientdet.efficientdet(num_classes=dataset_val.num_classes(), pretrained=True, phi=parser.scaling_compound)
	retinanet.load_state_dict(torch.load(parser.model, map_location=device))
	
	#retinanet = torch.load(parser.model)

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()

	retinanet.eval()

	unnormalize = UnNormalizer()

	def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	for idx, data in enumerate(dataloader_val):

		with torch.no_grad():
			st = time.time()
			scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
			print('Elapsed time: {}'.format(time.time()-st))
			idxs = np.where(scores>0.5)
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				draw_caption(img, (x1, y1, x2, y2), label_name)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
				print(label_name)
                        plt.imshow(img)
			plt.show()
			#gc.cv2_imshow(img)
			#cv2.waitKey(0)



if __name__ == '__main__':
 main()
