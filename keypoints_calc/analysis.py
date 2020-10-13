import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import KeypointsGauss
#from src.model_multi_headed import KeypointsGauss
from src.dataset import KeypointsDataset, transform
from src.prediction import Prediction
from datetime import datetime
from PIL import Image
import numpy as np

# model
keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
#keypoints.load_state_dict(torch.load('checkpoints/dr_braid_varied/model_2_1_5_0.0026437892680103254.pth'))
keypoints.load_state_dict(torch.load('checkpoints/black_out_simple_GAUSS_KPTS_ONLY/model_2_1_6_0.003018017431239951.pth'))
#keypoints.load_state_dict(torch.load('checkpoints/undo_reid_termGAUSS_KPTS_ONLY/model_2_1_4_0.003552900240372758.pth'))

#keypoints.load_state_dict(torch.load('checkpoints/undo_reid_term/model_2_1_13.pth'))
#keypoints.load_state_dict(torch.load('checkpoints/undo_reid_term/model_2_1_14.pth'))
#keypoints.load_state_dict(torch.load('checkpoints/undo_reid_term_braid_GAUSS_KPTS_ONLY/model_2_1_4_0.003553322965173738.pth'))
#keypoints.load_state_dict(torch.load('checkpoints/undo_reid_term_capsule_GAUSS_KPTS_ONLY/model_2_1_4_0.0032973564452098247.pth'))

# cuda
use_cuda = torch.cuda.is_available()
#use_cuda = False
if use_cuda:
    torch.cuda.set_device(7)
    keypoints = keypoints.cuda()

prediction = Prediction(keypoints, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, use_cuda)
transform = transform = transforms.Compose([
    transforms.ToTensor()
])

#image_dir = 'data/global_cable/images'
#image_dir = 'data/undo_reid_term_braid/test/images'
#image_dir = 'data/undo_reid_term_capsule/test/images'
image_dir = 'data/black_out_simple/test/blacked_out'
#image_dir = 'data/real_braid_1'
classes = {0: "Undo", 1:"Reidemeister", 2:"Terminate"}
for i, f in enumerate(sorted(os.listdir(image_dir))):
    img = cv2.imread(os.path.join(image_dir, f))
    #dim = (640,480)
    #img = cv2.resize(img,dim) 
    img_t = transform(img)
    img_t = img_t.cuda()
    # GAUSS
    heatmap = prediction.predict(img_t)
    heatmap = heatmap.detach().cpu().numpy()
    prediction.plot(img, heatmap, image_id=i)
 
    #heatmap, cls = prediction.predict(img_t)
    #cls = torch.argmax(cls).item()
    #heatmap = heatmap.detach().cpu().numpy()
    #prediction.plot(img, heatmap, image_id=i, cls=cls, classes=classes)