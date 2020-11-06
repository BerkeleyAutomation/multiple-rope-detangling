
import cv2
import os
import numpy as np

def crop_rope(img, crop_size, network_input_size, plot=True):
    vis = img.copy()
    
    # First, crop the image to a (fairly large) region containing the workspace
    wkspace_x, wkspace_y = (670,250) # Hardcoded pixels for where the workspace starts
    wkspace_w, wkspace_h = (560, 420) # Hardcoded workspace height/width
    workspace_crop_vis = cv2.rectangle(vis, (wkspace_x, wkspace_y), (wkspace_x+wkspace_w, wkspace_y+wkspace_h), (255,0,0), 1)
    workspace_crop = vis[wkspace_y:wkspace_y + wkspace_h, wkspace_x:wkspace_x + wkspace_w]

    # Mask out the rope, get the center of the rope
    workspace_gray = cv2.cvtColor(workspace_crop, cv2.COLOR_BGR2GRAY)
    _, workspace_mask = cv2.threshold(workspace_gray,127,255,cv2.THRESH_BINARY)
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
    previous_crop = img[360:360+final_crop_height, 775:775+final_crop_width]
    network_input = cv2.resize(resulting_crop, network_input_size)
    previous_crop = cv2.resize(previous_crop, network_input_size)
    
    if plot:
        cv2.circle(workspace_crop, rope_center, 5, (100,0,255), -1)
        cv2.imshow("img", workspace_crop_vis)
        cv2.waitKey(0)
        #cv2.imshow("img", workspace_crop)
        #cv2.waitKey(0)
        #cv2.imshow("img", workspace_mask)
        #cv2.waitKey(0)
        #cv2.imshow("img", resulting_crop)
        #cv2.waitKey(0)
        cv2.imshow("img", np.hstack((previous_crop, network_input)))
        #cv2.imshow("img", network_input)
        cv2.waitKey(0)
    return network_input, global_x, global_y

if __name__ == '__main__':
    image_dir = './real_images/hairtie'
    crop_size = (400,300)
    network_input_size = (640,480)
    folder = 'hairtie2'
    output_folder = folder + '_resized'
    for f in os.listdir(image_dir):
        if f != '.DS_Store':
            image_path = os.path.join(image_dir, f)
            img = cv2.imread(image_path)
            resized, _, _ = crop_rope(img, crop_size, network_input_size, plot=False)
            cv2.imwrite(os.path.join(output_folder, f), resized)