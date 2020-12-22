import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from image_utils import *
from grad_utils import *
from skimage.morphology import skeletonize
from skimage import data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage import img_as_ubyte

def preprocess(img):
    resized = cv2.resize(img.copy(), (640,480))
    cv2.imshow('vis', resized)
    cv2.waitKey(0) 
    
    # Inpaint, low-pass & threshold image
    inpainted = inpainted_binary(resized)
    blurred = blur(inpainted, (11,11))
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    cx, cy = locate_circle_center_hough(thresh)
    cv2.imshow('vis', thresh)
    cv2.waitKey(0) 
    # Skeletonize
    binary = thresh/255
    skeleton = skeletonize(binary)
    skeleton_cv = img_as_ubyte(skeleton)
    cv2.imshow('vis', skeleton_cv)
    cv2.waitKey(0) 

    vis = np.stack((skeleton_cv,)*3, axis=-1) # 3-channel version of skeleton

    circle_center = closest_rope(skeleton_cv, cx, cy)
    start_point = closest_rope(skeleton_cv, cx, cy, radius=40)
    cv2.circle(vis, tuple(start_point), 3, (255,0,0), -1)
    cv2.imshow('vis', vis) 
    cv2.waitKey(0) 
    return skeleton_cv, start_point, circle_center

def trace(image, circle_rad=40, debug_plot=False, annotation_plot=False):
    points = []
    img, start_point, circle_center = preprocess(image)
    cx, cy = circle_center
    ptx, pty = start_point

    vis = np.stack((img,)*3, axis=-1) # 3-channel version of img

    # DEBUGGING: Show found circle and radius to start looking outwards from 
    if debug_plot:
        cv2.circle(vis, (cx, cy), circle_rad, (0, 255, 0), 3)
        cv2.circle(vis, (ptx, pty), 3, (170, 0, 0), -1)
        cv2.imshow("vis", vis)
        cv2.waitKey(0)

    not_end_of_rope = True
    steps_taken = 0
    max_steps = 55
    symm_dist_thresh = 10
    triangle_area_thresh = 75
    while not_end_of_rope:
        points.append((ptx, pty))
        k = 2
        step_size = 17
        match1, match2 = closest_rope(img, ptx, pty, k=k, error_margin=0, radius=step_size)

        # This is a weird edge case where the skeleton is actually slightly more than 1px wide
        # This causes two neighbors to end up very close to each other (but they should be on either side of (ptx, pty)
        # When this happens, just get the next neighbor until you get two matches symmetric about the point of interest

        while dist(match1, match2) < symm_dist_thresh:
            k += 1
            matches = closest_rope(img, ptx, pty, k=k, radius=step_size)
            match2 = matches[-1]
        match = match1
        if steps_taken == 0:
            d1 = dist([match1[0], match1[1]], [ptx, pty])
            d2 = dist([match1[0], match1[1]], [cx, cy])
        else:
            d1 = dist([match1[0], match1[1]], points[steps_taken])
            d2 = dist([match1[0], match1[1]], points[steps_taken - 1])
        if d1 > d2:
            match = match2
        temp = vis.copy()

        triangle_area = cv2.contourArea(np.array([np.array(match1), np.array((ptx,pty)), np.array(match2)]))
        #print(triangle_area)

        # We have encountered a loop
        if triangle_area > triangle_area_thresh and len(points) > 1:
            last_ptx, last_pty = points[-2]
            dx = ptx - last_ptx
            dy = pty - last_pty
            new_match = closest_rope(img, ptx+dx, pty+dy, k=1)
            match = new_match
            if debug_plot:
                cv2.circle(temp, tuple(new_match), 3, (0, 255, 100), -1)

        # DEBUGGING: show each step of the trace, with the 2 candidate NN's for each point
        if debug_plot:
            cv2.circle(temp, tuple(match1), 3, (0, 255, 255), -1)
            cv2.circle(temp, (ptx, pty), 3, (0, 0, 255), -1)
            cv2.circle(temp, tuple(match2), 3, (255, 255, 0), -1)
            cv2.imshow("vis", temp)
            cv2.waitKey(0)

        if steps_taken > max_steps:
            not_end_of_rope = False
        if annotation_plot:
            if steps_taken % 1 == 0: # Change 1 to however often you want to trace
                k = cv2.waitKey(0) % 256
                if k == ord('e'):
                    print "pressed e, ending"
                    break
                if k == ord('r'):
                    print "pressed r, restarting"
                    points = []
                    ptx, pty = start_point
                    steps_taken = 0
                    continue
                else:
                    plot_points(img, points)
        ptx, pty = match
        steps_taken += 1
    return points

if __name__ == '__main__':
    for i in range(20):
        img = cv2.imread('../datasets/easy/%06d_rgb.png'%i, 0)
        #img = cv2.imread('../datasets/hard/phoxi/segdepth_%d.png'%i, 0)
        _,img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        #points = trace(img)
        points = trace(img, debug_plot=True)
        resized = cv2.resize(img, (640,480))
        plot_points(resized, points)