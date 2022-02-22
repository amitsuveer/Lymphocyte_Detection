import os
import os.path as osp
import numpy as np    
import pandas as pd
import tqdm


def makeDirectory(path_to_directory:str=''):    
    from pathlib import Path
    p = Path(path_to_directory)
    p.mkdir(exist_ok=True, parents=True)



def create_io_dirs(root_dir:str='.', data_dir:str=''):
    
    #org_data_dir = osp.join(root_dir, 'lymphocyte')
    org_data_dir = data_dir
    in_img_path = osp.join(org_data_dir, 'data')
    in_msk_path = osp.join(org_data_dir, 'manual_seg')

    ext_img_path = osp.join(root_dir, 'setup', 'data_ext')
    ext_msk_path = osp.join(root_dir, 'setup', 'cent_ext')

    # make output directories
    makeDirectory(ext_img_path)
    makeDirectory(ext_msk_path)
    
    # YOLO Directory structure
    image_train_path = osp.join(root_dir, 'images', 'train')
    image_val_path = osp.join(root_dir, 'images', 'val')
    image_test_path = osp.join(root_dir, 'images', 'test')

    labels_path_train = osp.join(root_dir, 'labels', 'train')
    labels_path_val = osp.join(root_dir, 'labels', 'val')
    labels_path_test = osp.join(root_dir, 'labels', 'test')


    makeDirectory(image_train_path)
    makeDirectory(image_val_path)
    makeDirectory(image_test_path)

    makeDirectory(labels_path_train)
    makeDirectory(labels_path_val)
    makeDirectory(labels_path_test)
    
    
    pred_mask_path = osp.join(root_dir, 'results', 'masks_predicted')
    masks_comp_path = osp.join(root_dir, 'results', 'masks_comparison')
    masks_img_comp_path = osp.join(root_dir, 'results', 'masks_img_comparison')
    qual_eval_path = osp.join(root_dir, 'results', 'comparison_images_masks')
    
    makeDirectory(pred_mask_path)        
    makeDirectory(masks_comp_path)
    make_mask_name(masks_img_comp_path)
    make_mask_name(qual_eval_path)
    
    perf_plots_path = osp.join(root_dir, 'results', 'performance_plots')
    make_mask_name(perf_plots_path)
    
    # predicted labels path -- auto-generated by script
    #pred_lab_path_test = 'yolov5/runs/detect/yolov5s_lymphocyte/labels' 
    #pred_lab_path_val = 'yolov5/runs/detect/yolov5s_lymphocyte2/labels' 
    #pred_lab_path_train = 'yolov5/runs/detect/yolov5s_lymphocyte3/labels' 
    #pred_lab_path_org = 'yolov5/runs/detect/yolov5s_lymphocyte4/labels' 
    pred_lab_path_org = 'yolov5/runs/detect/yolov5s_lymphocyte/labels' 
    
                #'pred_labels_test' : pred_lab_path_test,
                #'pred_labels_valid': pred_lab_path_val,
                #'pred_labels_train': pred_lab_path_train,
    
    io_paths = {'original_data'    : org_data_dir,
                'original_images'  : in_img_path,
                'original_centers' : in_msk_path,
                'extended_images'  : ext_img_path,
                'extended_centers' : ext_msk_path,
                'images_train'     : image_train_path,
                'images_valid'     : image_val_path, 
                'images_test'      : image_test_path,
                'labels_train'     : labels_path_train,
                'labels_valid'     : labels_path_val, 
                'labels_test'      : labels_path_test,
                'pred_labels_org'  : pred_lab_path_org,
                'pred_masks'       : pred_mask_path,
                'comp_masks'       : masks_comp_path,
                'masks_img_comp'   : masks_img_comp_path,
                'subjective_eval'  : qual_eval_path,
                'perf_plots'       : perf_plots_path
                }
    
    return io_paths


    
def write_images(info_dict:dict, op_path:str):
    import os
    import os.path as osp
    for im_name, im_data in info_dict.items():
        im_data.save(osp.join(op_path, im_name))



def make_mask_name(im_name:str):
    idx = im_name.find('_')
    if idx>0:
        # '_' found, the image is augmented image
        ms_name = f'{im_name[2:idx]}m{im_name[idx:]}'
    else:
        # '_' not found, image is original image, seek for '.'
        idx = im_name.find('.')
        ms_name = f'{im_name[2:idx]}m{im_name[idx:]}'
    return ms_name



#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    import shutil
    for f in tqdm.tqdm(list_of_files):
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False
            
            
#Utility function to move images 
def copy_files_to_folder(list_of_files, destination_folder):
    import shutil
    for f in tqdm.tqdm(list_of_files):
        try:
            shutil.copy(f, destination_folder)
        except:
            print(f)
            assert False

def display_pair(img, msk):
    import matplotlib.pyplot as plt
        
    plt.figure(figsize=(16,16))
    plt.subplot(1,2,1)
    plt.imshow(img, interpolation='none')
    plt.subplot(1,2,2)
    plt.imshow(img, interpolation='none')
    plt.imshow(msk, 'jet', interpolation='none', alpha=0.7)
    plt.show()
                