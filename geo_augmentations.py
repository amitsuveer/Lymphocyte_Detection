import os
import os.path as osp
import tqdm 
import numpy as np
from scipy import ndimage
from PIL import Image    
from utils import write_images
  
def augment_basic(img:np.array, name:str):
    
    name_img_dict = {}
    
    name_img_dict[name] = Image.fromarray(img) # Original Image
    
    img_90 = ndimage.rotate(img, 90) # Rotate 90
    name_img_dict[f'{name[:-4]}_R1{name[-4:]}'] = Image.fromarray(img_90)
    
    img_180 = ndimage.rotate(img, 180) # Rotate 180
    name_img_dict[f'{name[:-4]}_R2{name[-4:]}'] = Image.fromarray(img_180)
    
    img_270 = ndimage.rotate(img, 270) # Rotate 270
    name_img_dict[f'{name[:-4]}_R3{name[-4:]}'] = Image.fromarray(img_270)
    
    img_FH = np.fliplr(img) # Flip Horizontal
    name_img_dict[f'{name[:-4]}_FH{name[-4:]}'] = Image.fromarray(img_FH)
    
    img_FV = np.flipud(img) # Flip Verticle
    name_img_dict[f'{name[:-4]}_FV{name[-4:]}'] = Image.fromarray(img_FV)
    
    return name_img_dict
    #return img_90, img_180, img_270, img_FH, img_FV


def geometrical_augmentations(io_paths:dict, test_FileNames:list):
    
    allFileNames = os.listdir(io_paths['original_images'])
    total_images = len(allFileNames)
    
    for i in tqdm.tqdm(range(1,total_images+1)):
        
        im_name = f'im{i}.tif'
        ms_name = f'{i}m.tif'
        
        # If image is in test data -- do NOT augment
        if im_name in test_FileNames:
            continue
        
        # read image & centre-location mask
        img = np.array(Image.open(osp.join(io_paths['original_images'], im_name)))
        msk = np.array(Image.open(osp.join(io_paths['original_centers'], ms_name)))
        
        # perform basic augmentations: R90, R180, R270, F-H, F-V
        name_data_img_dict = augment_basic(img=img, name=im_name)
        name_data_msk_dict = augment_basic(img=msk, name=ms_name)
        
        # create name-data dictionary
        write_images(info_dict=name_data_img_dict, op_path=io_paths['extended_images'])
        write_images(info_dict=name_data_msk_dict, op_path=io_paths['extended_centers'])

        del img, msk, name_data_img_dict, name_data_msk_dict
    
