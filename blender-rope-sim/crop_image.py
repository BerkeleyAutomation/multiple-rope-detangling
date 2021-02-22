import os
import cv2
import numpy as np


def crop_and_resize(img, aspect=(80,60)):
	h, w, channels = img.shape
	x1, y1, x2, y2 = 0, 0, w-1, h-1
	x_min, x_max = min(x1,x2), max(x1,x2)
	y_min, y_max = min(y1,y2), max(y1,y2)
	box_width = x_max - x_min
	box_height = y_max - y_min
	# resize this crop to be 320x240
	new_width = int((box_height*aspect[0])/aspect[1])
	offset = new_width - box_width
	x_min -= int(offset/2)
	x_max += offset - int(offset/2)
	crop = img[y_min:y_max, x_min:x_max]
	resized = cv2.resize(crop, aspect)
	rescale_factor = new_width/aspect[0]
	offset = (x_min, y_min)
	return resized, rescale_factor, offset


def crop_at_point(img, x, y, width=640, height=480):
	img = img[y:y+height, x:x+width]
	return img


if __name__ == '__main__':
	input_pre = './real_images/original_lab_images/'
	folder = 'lab_two_rope'
	input_folder = input_pre + folder
	#output_pre = './real_images/cropped/'
	output_folder = folder + '_resized'
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)
	for f in os.listdir(input_folder):
		print(f)
		img = cv2.imread(os.path.join(input_folder, f))
		#resized, _, _ = crop_and_resize(img, aspect=(286,286))
		#resized = crop_at_point(img, 740, 200, width=480, height=480)
		# resized = crop_at_point(img, 700, 200, width=640, height=480)
		# resized = crop_at_point(img, 850, 325, width=450, height=400)
		# resized = crop_at_point(img, 200, 1200, width=2500, height=2000)
		resized = crop_at_point(img, 800, 275, width=450, height=400)
		# resized = crop_at_point(img, 875, 250, width=450, height=400)
		resized = cv2.resize(resized, (640,480))
		cv2.imwrite(os.path.join(output_folder, f), resized)
		#cv2.imshow('resized', resized)
		#cv2.waitKey(0)

