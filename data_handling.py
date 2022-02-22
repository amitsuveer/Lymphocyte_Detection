import os
import os.path as osp
import numpy as np
import pandas as pd
import tqdm
from PIL import Image

from utils import make_mask_name
from utils import move_files_to_folder, copy_files_to_folder

def split_test_set(ratio:float=0.10, paths:dict=None):
    
    allFileNames = os.listdir(paths['original_images'])
    np.random.shuffle(allFileNames)

    total_files = int(len(allFileNames))
    _, test_FileNames = np.split(np.array(allFileNames), [int(total_files * (1-ratio))])
    
    return test_FileNames
    
    
def split_train_valid_set(train_ratio:float=0.80, paths:dict=None):
    
    tune_ratio = 1.0 - train_ratio

    extended_FileNames = os.listdir(paths['extended_images'])
    np.random.shuffle(extended_FileNames)
    train_FileNames, tune_FileNames = np.split(extended_FileNames, [int(len(extended_FileNames) * (1-tune_ratio))])
    
    return train_FileNames, tune_FileNames


def move_data(io_paths:dict, train_files:list, valid_files:list, test_files:list):
    
    # get the input path
    ext_img_path = io_paths['extended_images']
    train_filepaths = [osp.join(ext_img_path, fname) for fname in train_files]
    move_files_to_folder(train_filepaths, io_paths['images_train'])
    
    valid_filepaths = [osp.join(ext_img_path, fname) for fname in valid_files]
    move_files_to_folder(valid_filepaths, io_paths['images_valid'])

    # Note that we copy test images, since we want to keep the original test images
    org_img_path = io_paths['original_images']
    test_filepaths = [osp.join(org_img_path, fname) for fname in test_files]
    copy_files_to_folder(test_filepaths, io_paths['images_test'])


def create_info_dataframes(io_paths:dict, train_files:list, valid_files:list, test_files:list):
    import os.path as osp
    
    train_df = create_pandas_dataframe(fileNames=train_files, tag='train', img_path=io_paths['images_train'], msk_path=io_paths['extended_centers'])
    tune_df = create_pandas_dataframe(fileNames=valid_files, tag='valid', img_path=io_paths['images_valid'], msk_path=io_paths['extended_centers'])
    test_df = create_pandas_dataframe(fileNames=test_files, tag='test', img_path=io_paths['images_test'], msk_path=io_paths['original_centers'])

    # Concate dataframes and write a single info dataframe
    df_all = pd.concat([test_df, tune_df, train_df], ignore_index=True)
    
    # write train-tune-test info to respective folder in csv format
    df_all.to_csv('data_info.csv')

    return df_all


def create_pandas_dataframe(fileNames:list, tag:str, img_path:str, msk_path:str):
    import os.path as osp
    
    temp_LOD = []
    for fname in fileNames:
        
        im_name = fname
        ms_name = make_mask_name(im_name)
                
        im_fullpath = osp.join(img_path, im_name)
        ms_fullpath = osp.join(msk_path, ms_name)
        
        # retrive origin: original or augmented        
        origin = 'aug' if im_name.find('_')>0 else 'org'
        
        temp_dict = {'name': im_name, 'origin':origin, 'data':tag, 
                     'image_fullpath': im_fullpath, 'mask_fullpath': ms_fullpath}
        
        temp_LOD.append(temp_dict)
    
    op_df = pd.DataFrame(temp_LOD)
    
    return op_df


def generate_YOLO_annotations(df_all:pd.DataFrame, io_paths:dict, bb_radii:int):
    
    # create YOLO annotations from the original data
    df_train = (df_all[df_all['data']=='train']).reset_index(drop=True).copy(deep=True)
    lab_paths_train = create_yolov5_annotations(df_name=df_train, op_filepath=io_paths['labels_train'], bb_radii=bb_radii)
    df_train['labels_fullpath'] = lab_paths_train
    
    df_valid = (df_all[df_all['data']=='valid']).reset_index(drop=True).copy(deep=True)
    lab_paths_val = create_yolov5_annotations(df_name=df_valid, op_filepath=io_paths['labels_valid'], bb_radii=bb_radii)
    df_valid['labels_fullpath'] = lab_paths_val
    
    df_test = (df_all[df_all['data']=='test']).reset_index(drop=True).copy(deep=True)
    lab_paths_test = create_yolov5_annotations(df_name=df_test, op_filepath=io_paths['labels_test'], bb_radii=bb_radii)
    df_test['labels_fullpath'] = lab_paths_test
    
    df_all = pd.concat([df_test, df_valid, df_train], ignore_index=True)
    
    return df_all 

def create_yolov5_annotations(df_name:pd.DataFrame, op_filepath:str, bb_radii:int=6, class_id:int=0):

    label_fullpaths = []
    
    #for idx, row in df_name.iterrows():
    for idx in range(len(df_name.index)):
        
        im_name = df_name.at[idx, 'name']
        lb_name = im_name.replace('tif', 'txt')
                
        info_buffer = []

        # read mask -- basically have object centers
        msk = np.array(Image.open(df_name.at[idx, 'mask_fullpath']))

        msk = msk[:,:,1]
        im_h, im_w = msk.shape[0],msk.shape[1] 
        
        obj_cnt = 0
        for row in range(im_h):
            for col in range(im_w):
            
                if msk[row, col]:

                    obj_cnt += 1
                    
                    # bbox co-ordinates as per the format required by YOLO v5                
                    ### centers
                    b_cent_x = col+1
                    b_cent_y = row+1

                    ### box limits (for the corner points)
                    bb_x_min = max(0, b_cent_x-bb_radii)
                    bb_x_max = min(im_w, b_cent_x+bb_radii)

                    bb_y_min = max(0, b_cent_y-bb_radii)
                    bb_y_max = min(im_h, b_cent_y+bb_radii)
                    
                    ### box width and height 
                    b_width = bb_x_max - bb_x_min
                    b_height = bb_y_max - bb_y_min
                    
                    # Normalise the co-ordinates by the dimensions of the image
                    b_cent_x /= im_w
                    b_cent_y /= im_h
                    b_width /= im_w
                    b_height /= im_h
                    
                    info_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_cent_x, b_cent_y, b_width, b_height))

              
                
        # Name of the file which we have to save 
        label_filename = osp.join(op_filepath, lb_name)
                
        # Save the annotation to disk
        print("\n".join(info_buffer), file=open(label_filename, "w"))
        
        label_fullpaths.append(label_filename)
    
    return label_fullpaths 
