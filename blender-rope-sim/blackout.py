import cv2
import numpy as np
import argparse
from xml.etree import ElementTree
import os
import math
import json
import colorsys
from random import randint

def blackout(idx, version, kpt_dir):
	image_filename = "{0:05d}_{1:01d}.jpg".format(idx, version)
	img = cv2.imread('render_kpts_output/extracted_images/{}'.format(image_filename))
	vis = img.copy()
	kpts = np.load('%s/%05d_%01d.npy'%(kpt_dir, idx, version))
	for i, (u,v) in enumerate(kpts):  
		#randomize the dropout shapes, sizes, and rotations 
		rect_or_circle = randint(0,1) 
		if rect_or_circle:
			(W,H) = (10,15)
			ang = randint(-60, 60)
			P0 = (u,v-5)
			rr = RRect(P0,(W,H),ang)
			rr.draw(vis)
		else:
			cv2.circle(vis,(u,v),12,(0,0,0), -1)
	blacked_out_filename = "{0:05d}_{1:01d}_blacked_out.png".format(idx, version)
	print("Blacking out: %d, %d"% (idx, version))
	cv2.imwrite('./render_kpts_output/blacked_out/{}'.format(blacked_out_filename), vis)

class RRect:
	def __init__(self, p0, s, ang):
	    self.p0 = [int(p0[0]),int(p0[1])]
	    (self.W, self.H) = s
	    self.ang = ang
	    self.p1,self.p2,self.p3 = self.get_verts(p0,s[0],s[1],ang)
	    self.verts = [self.p0,self.p1,self.p2,self.p3]

	def get_verts(self, p0, W, H, ang):
		sin = np.sin(ang/180*3.14159)
		cos = np.cos(ang/180*3.14159)
		P1 = [int(self.H*sin)+p0[0],int(self.H*cos)+p0[1]]
		P2 = [int(self.W*cos)+P1[0],int(-self.W*sin)+P1[1]]
		P3 = [int(self.W*cos)+p0[0],int(-self.W*sin)+p0[1]]
		return [P1,P2,P3]

	def draw(self, image):
		for i in range(len(self.verts)-1):
			cv2.line(image, (self.verts[i][0], self.verts[i][1]), (self.verts[i+1][0],self.verts[i+1][1]), (0,0,0), 9)
		cv2.line(image, (self.verts[3][0], self.verts[3][1]), (self.verts[0][0], self.verts[0][1]), (0,0,0), 9)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir', type=str, default='render_kpts_output/black_out_pixels')
	args = parser.parse_args()
	if not os.path.exists("./render_kpts_output/blacked_out"):
		os.makedirs('./render_kpts_output/blacked_out')
	else:
		os.system("rm -rf ./render_kpts_output/blacked_out")
		os.makedirs("./render_kpts_output/blacked_out")
	for i in range(len(os.listdir(args.dir))//3):
		for j in range(3):
			blackout(i, j, args.dir)