import sys
sys.path.append('segmentation_models/U-2-Net')

import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir,imsize=512):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    # img_name = image_name.split(os.sep)[-1]
    # image = io.imread(image_name)
    imo = im.resize((imsize,imsize),resample=Image.BILINEAR)

    imo.save(os.path.join(d_dir, image_name + '.png'))

def get_saliency_maps(pil_images:list):
    
    model_name='u2net'#u2netp

    model_dir = os.path.join(os.getcwd(), 'segmentation_models', 'U-2-Net', 'saved_models', model_name, model_name + '.pth')
    images = []
    for i, pil_image in enumerate(pil_images):
        input_image = np.array(pil_image)
        label = np.zeros(input_image.shape)[:,:,0]
        label = label[:,:,np.newaxis]
        input_dict = {'imidx':np.array([i]), 'image':input_image, 'label':label}
        out_dict = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])(input_dict)
        images.append(out_dict["image"])
    images = torch.stack(images)

    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    # elif(model_name=='u2netp'):
    #     print("...load U2NEP---4.7 MB")
    #     net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    inputs_test = images
    inputs_test = inputs_test.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

    # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)
    
    # save results to test_results folder
    # prediction_dir = "temp"
    # if not os.path.exists(prediction_dir):
    #     os.makedirs(prediction_dir, exist_ok=True)
    # for i, each_pred in enumerate(pred):
    #     save_output(f"saliency_map_{i}",each_pred,prediction_dir)

    # del d1,d2,d3,d4,d5,d6,d7

    return pred