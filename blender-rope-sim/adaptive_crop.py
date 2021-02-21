import cv2
import os
import numpy as np

def crop_rope(img, x, y, crop_size, plot=True):
    vis = img.copy()

    # First, crop the image to a (fairly large) region containing the workspace
    wkspace_x, wkspace_y = (x,y) # Hardcoded pixels for where the workspace starts
    wkspace_w, wkspace_h = (470, 480) # Hardcoded workspace height/width
    workspace_crop_vis = cv2.rectangle(vis, (wkspace_x, wkspace_y), (wkspace_x+wkspace_w, wkspace_y+wkspace_h), (255,0,0), 1)
    workspace_crop = vis[wkspace_y:wkspace_y + wkspace_h, wkspace_x:wkspace_x + wkspace_w]

    # Mask out the rope, get the center of the rope
    hsv = cv2.cvtColor(workspace_crop, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0,0,0])
    upper_black = np.array([179,255,127])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    workspace_mask = cv2.bitwise_not(mask)
    #workspace_gray = cv2.cvtColor(workspace_crop, cv2.COLOR_BGR2GRAY)
    #_, workspace_mask = cv2.threshold(workspace_gray,127,255,cv2.THRESH_BINARY)
    mask_idxs = np.nonzero(workspace_mask)
    ys, xs = mask_idxs
    mu_y = int(np.mean(ys))
    mu_x = int(np.mean(xs))
    rope_center = (mu_x, mu_y)

    # Take a crop centered at the rope, and upscale the image
    final_crop_width, final_crop_height = crop_size
    global_x = wkspace_x + mu_x - final_crop_width//2
    global_y = wkspace_y + mu_y - final_crop_height//2
    resulting_crop = img[global_y:global_y+final_crop_height, global_x:global_x+final_crop_width]
    box = [global_x, global_y, global_x+final_crop_width, global_y+final_crop_height]
    previous_crop = img[360:360+final_crop_height, 775:775+final_crop_width]

    if plot:
        cv2.circle(workspace_crop, rope_center, 5, (100,0,255), -1)
        cv2.imshow("img", workspace_crop_vis)
        cv2.waitKey(0)
    return box

def crop_and_resize(box, img, aspect=(640,480)):
    x1, y1, x2, y2 = box
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

if __name__ == '__main__':
    #image_dir = './real_images/original_lab_images/lab_terminated'
    image_dir = './real_images/original_lab_images/two_white'
    output_folder = image_dir + '_crop'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    #crop_size = (500,420)
    #crop_size = (430,320)
    crop_size = (400,300)
    network_input_size = (640,480)
    for f in os.listdir(image_dir):
        if f != '.DS_Store':
            print(f)
            image_path = os.path.join(image_dir, f)
            img = cv2.imread(image_path)
            x, y = 850, 250
            box = crop_rope(img, x, y, crop_size, plot=False)
            resized, _, _ = crop_and_resize(box, img)
            cv2.imwrite(os.path.join(output_folder, f), resized)
