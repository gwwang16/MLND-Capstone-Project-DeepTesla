
import cv2
import params
import pandas as pd
import numpy as np
import os

img_height = params.img_height
img_width = params.img_width
img_channels = params.img_channels

data_dir = params.data_dir
out_dir = params.out_dir
model_dir = params.model_dir


def preprocess(img):
	'''resize and crop the image
	:img: The image to be processed
	:return: Returns the processed image'''
	# Chop off 1/2 from the top and cut bottom 150px(which contains the head of car)
	ratio = img_height / img_width
	h1, h2 = int(img.shape[0]/2),img.shape[0]-150
	w = (h2-h1) / ratio
	padding = int(round((img.shape[1] - w) / 2))
	img = img[h1:h2, padding:-padding]
	## Resize the image
	img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
	## Image Normalization
	#img = img / 255.
	return img

def frame_count_func(file_path):
	'''return frame count of this video'''
	cap = cv2.VideoCapture(file_path)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	return frame_count

def load_data(mode):
	'''get train or valid batch data,
	mode: train or valid,
	output: batch data.'''
	if mode == 'train':
		epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	#elif mode == 'valid':
	#	epochs = [1, 2, 7, 9]
	elif mode == 'test':
		epochs = [10]
	else:
		print('Wrong mode input')
		
	imgs = []
	wheels = []
	# extract image and steering data
	for epoch_id in epochs: 
		yy=[]
		
		vid_path = os.path.join(data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
		frame_count = frame_count_func(vid_path)
		cap = cv2.VideoCapture(vid_path)
				  
		csv_path = os.path.join(data_dir, 'epoch{:0>2}_steering.csv'.format(epoch_id))
		rows = pd.read_csv(csv_path)
		yy = rows['wheel'].values
		wheels.extend(yy)
		
		while True:
			ret, img = cap.read()
			if not ret:
				break
			img = preprocess(img)
			imgs.append(img)
		
		assert len(imgs) == len(wheels)

		cap.release()
			
	return imgs, wheels


def load_batch(imgs, wheels):

    assert len(imgs) == len(wheels)
    n = len(imgs)

    assert n > 0

    ii = random.sample(range(0, n), params.batch_size)
    assert len(ii) == params.batch_size

    xx, yy = [], []
    for i in ii:
        xx.append(imgs[i])
        yy.append(wheels[i])

    return xx, yy