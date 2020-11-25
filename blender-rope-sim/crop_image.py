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
	input_pre = './real_images/'
	folder = 'square_pretzel_fig8'
	input_folder = input_pre + folder
	#output_pre = './real_images/cropped/'
	output_folder = folder + '_resized'
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)
	for f in os.listdir(input_folder):
		img = cv2.imread(os.path.join(input_folder, f))
		#resized, _, _ = crop_and_resize(img, aspect=(286,286))
		#resized = crop_at_point(img, 740, 200, width=480, height=480)
		#resized = crop_at_point(img, 700, 200, width=640, height=480)
		resized = crop_at_point(img, 850, 325, width=450, height=400)
		resized = cv2.resize(resized, (640,480))
		cv2.imwrite(os.path.join(output_folder, f), resized)
		#cv2.imshow('resized', resized)
		#cv2.waitKey(0)

# import cv2
# import os
# import sys
# import argparse
# import numpy as np
# from shutil import copyfile
# import json

# def crop_resize(img, filename):
# 	dim = (640,480)
# 	new_img = img[200:700, 600:1250]
# 	new_img = cv2.resize(new_img, dim)
# 	cv2.imwrite('./real_images/cropped/cropped_hairtie/{}'.format(filename), new_img)

# def check_size(img):
# 	print(img.shape)


# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('-d', '--dir', type=str, default='./real_images/original/overhead_hairtie')
# 	args = parser.parse_args()
# 	if os.path.exists('./real_images/cropped/cropped_hairtie'):
# 		os.system('rm -r ./real_images/cropped/cropped_hairtie')
# 	os.mkdir('./real_images/cropped/cropped_hairtie')
# 	img_dir = args.dir
# 	for i in range(1, len(os.listdir(img_dir)) + 1):
# 		img = cv2.imread(os.path.join(img_dir, 'color%d.png'%i))
# 		filename = '%05d.jpg'%i
# 		crop_resize(img, filename)


