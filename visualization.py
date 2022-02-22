from evaluation import get_cent_mask_from_yolo_labels
from utils import makeDirectory
from PIL import Image
import pandas as pd
import numpy as np
import os.path as  osp
import seaborn as sns

import matplotlib.pyplot as plt



def display_pair(img, msk, scale=10):
    plt.figure(figsize=(scale,scale))
    # display original image
    plt.subplot(1,2,1)
    plt.imshow(img, interpolation='bicubic')
    plt.axis('off')
    
    # display image >> mask
    plt.subplot(1,2,2)
    plt.imshow(img, interpolation='bicubic')
    plt.axis('off')
    plt.imshow(msk, 'jet', interpolation='none', alpha=0.7)
    plt.axis('off')
    plt.show()

def display_trio(img, msk1, msk2, scale=10, alpha=0.5):      
    
    # display original image
    plt.figure(figsize=(scale,scale))
    plt.subplot(1,2,1)
    plt.imshow(img, interpolation='bicubic')
    plt.axis('off')
    
    # display image >> GT-mask >> Pred-mask
    plt.subplot(1,2,2)
    plt.imshow(img, interpolation='bicubic')
    plt.axis('off')
    plt.imshow(msk1, 'jet', interpolation='none', alpha=alpha)
    plt.axis('off')
    plt.imshow(msk2, 'jet', interpolation='none', alpha=alpha)
    plt.axis('off')    
    plt.show()

def perform_subjective_evaluation(df_all:pd.DataFrame, io_paths:dict):
    
    tags = ['train', 'valid', 'test', 'org']

    LOD = []
    for tag in tags:
        if tag=='org':
            df_name = df_all[(df_all['origin']=='org')].reset_index(drop=True).copy(deep=True)
        else:
            df_name = df_all[ (df_all['data']==tag) & (df_all['origin']=='org')].reset_index(drop=True).copy(deep=True)
        
        # perform subjective evaluation
        save_qualitative_results(df_name=df_name, paths=io_paths, data_tag=tag)
        
        # cleaning memory
        del df_name
    
    # Display info to the user
        
    print(f'\nPredicted masks saved at:\n {io_paths["pred_masks"]}.\n')
    
    print(f'Predicted & GT center comparison saved at:\n {io_paths["comp_masks"]}.\n')
    
    print(f'Predicted & GT center image overlay saved at at:\n  {io_paths["masks_img_comp"]}.\n')
    
    print(f'Subjective results saved at at:\n {io_paths["subjective_eval"]}.\n')



def save_qualitative_results(df_name:pd.DataFrame, paths:dict, data_tag:str):
    
    def make_op_dirs(paths:dict, data_tag:str):
        
        pred_mask_path = osp.join(paths['pred_masks'], data_tag)    
        makeDirectory(pred_mask_path)

        comp_masks_path = osp.join(paths['comp_masks'], data_tag)    
        makeDirectory(comp_masks_path)

        masks_img_comp_path = osp.join(paths['masks_img_comp'], data_tag)    
        makeDirectory(masks_img_comp_path)

        subj_eval_path = osp.join(paths['subjective_eval'], data_tag)    
        makeDirectory(subj_eval_path)
        
        op_path_dict = {'pred_masks': pred_mask_path,
                       'comp_masks': comp_masks_path,
                       'masks_img_comp': masks_img_comp_path,
                       'subjective_eval': subj_eval_path}
        
        return op_path_dict
    
    
    def save_predicted_mask(mask:np.array, fullpath:str):
        im_h, im_w, im_ch = mask.shape
        op_msk = 255*np.ones([im_h, im_w, 3], dtype=np.uint8)        
        # set out two channels to zero        
        op_msk[:,:,1] = op_msk[:,:,1] - mask[:,:,0] # re-setting green channel
        op_msk[:,:,2] = op_msk[:,:,2] - mask[:,:,0]# re-setting blue channel
        # save image
        (Image.fromarray(op_msk)).save(fullpath)
        #(Image.fromarray(op_msk)).save(osp.join(op_path_pred_msk, msk_fullpath.split('/')[-1]))
        
    
    def save_masks_overlays(gt_mask:np.array, pr_mask:np.array, fullpath:str):
        im_h, im_w, _ = pr_mask.shape
        op_msk = 255*np.ones([im_h, im_w, 3], dtype=np.uint8)
        # re-setting channels -- blue
        op_msk[:,:,2] = op_msk[:,:,2] - pr_mask[:,:,0] # remove R of pred-mask
        op_msk[:,:,2] = op_msk[:,:,2] - gt_mask[:,:,1] # remove  G of gt mask
        # re-setting Green channel
        op_msk[:,:,1] = op_msk[:,:,1] - pr_mask[:,:,0] # # remove R of pred-mask
        # re-setting Red channel
        op_msk[:,:,0] = op_msk[:,:,0] - gt_mask[:,:,1] # # remove R of pred-mask        
        # save image
        (Image.fromarray(op_msk)).save(fullpath)    
    
    
    def save_pred_gt_img_overlay(img:np.array, gt_mask:np.array, pr_mask:np.array, fullpath:str):
        im_h, im_w, _ = img.shape
        ov_img = np.ones([im_h, im_w, 3], dtype=np.uint8)
        for ch in range(3):
            ov_img[:,:,ch] = img[:,:,ch]
        
        for x in range(im_w):
            for y in range(im_h):
                # Set R-channel from Pred mask                
                if pr_mask[y,x,0]==255:
                    ov_img[y,x,0]=255
                    ov_img[y,x,1]=0
                    ov_img[y,x,2]=0
                    
                # Set G-channel from GT mask                
                if gt_mask[y,x,1]==255:
                    ov_img[y,x,0]=0 if ov_img[y,x,0]!=255 else 255
                    ov_img[y,x,1]=255
                    ov_img[y,x,2]=0        

        # save image
        (Image.fromarray(ov_img)).save(fullpath)        
        return ov_img
    
    
    def save_subjective_comp(org_img:np.array, overlay_img:np.array, fullpath:str):
        
        im_h, im_w, _ = org_img.shape
        
        pt_w = 10
        im_patch = 255 * np.ones([im_h,pt_w], dtype=np.uint8)        
        
        img = np.zeros([im_h, (2*im_w + pt_w), 3], dtype=np.uint8)
        
        for ch in range(3):
            img[:,:,ch] = np.hstack((org_img[:,:,ch], im_patch, overlay_img[:,:,ch]))
        
        # save image
        (Image.fromarray(img)).save(fullpath)
        
                
    # parent function code starts here!    
    
    # setting IO-paths
    op_path_dict = make_op_dirs(paths=paths, data_tag=data_tag)

    # Iterate over images
    for idx in range(len(df_name.index)):
        
        im_name = df_name.at[idx, 'name']
        img_fullpath = df_name.at[idx, 'image_fullpath']
        msk_fullpath = df_name.at[idx, 'mask_fullpath']

        # read original image and mask
        img = np.array(Image.open(img_fullpath))
        msk_gt = np.array(Image.open(msk_fullpath))
                
        # read predicted labels
        label_key = 'pred_labels_org' # Hard coding for now (can be customized)
        labels_fullpath = osp.join(paths[label_key],im_name.replace('tif', 'txt'))
        im_h, im_w = img.shape[0], img.shape[1]
        
        # generate predicted mask (centre locations)
        msk_pred = get_cent_mask_from_yolo_labels(bb_path=labels_fullpath, im_width=im_w, im_height=im_h)
        
        # save the predicted mask first          
        op_fullpath = osp.join(op_path_dict['pred_masks'],msk_fullpath.split('/')[-1])
        save_predicted_mask(mask=msk_pred, fullpath=op_fullpath)
        
        # save GT and Pred mask overlay
        op_fullpath = osp.join(op_path_dict['comp_masks'],msk_fullpath.split('/')[-1])
        save_masks_overlays(gt_mask=msk_gt, pr_mask=msk_pred, fullpath=op_fullpath)
        
        # Save GT and Pred mask overlay on Image   
        op_fullpath = osp.join(op_path_dict['masks_img_comp'],img_fullpath.split('/')[-1])
        img_overlay = save_pred_gt_img_overlay(img=img, gt_mask=msk_gt, pr_mask=msk_pred, fullpath=op_fullpath)
        
        # Save subjective evaluation images
        op_fullpath = osp.join(op_path_dict['subjective_eval'],img_fullpath.split('/')[-1])
        save_subjective_comp(org_img=img, overlay_img=img_overlay, fullpath=op_fullpath)
                            
    
'''
def save_confusion_matrix(perf_dict:dict, title:str, fullpath:str='', img_cnt:int=1, axs=None, ax_cnt=None):
    plot_confusion_matrix(perf_dict=perf_dict, title=title, img_cnt=img_cnt, is_save=True,
                          fullpath=fullpath, axs=axs, ax_cnt=ax_cnt)
 '''
        
def plot_confusion_matrix(perf_dict:dict, title:str, img_cnt:int=1, is_save:bool=False, fullpath:str='', axs=None, ax_cnt=None):
        
    sns.set(font_scale = 1.15)
    
    TP = perf_dict['TP']
    FP = perf_dict['FP']
    FN = perf_dict['FN']
    TN = perf_dict['TN']
    
    cf_matrix = [[TP, FN],[FP, TN]]
    cf_mat_flat = [TP, FN, FP, TN]

    group_names = ['TP:','FN:','FP:','TN:']

    group_counts = ["{0:0.0f}".format(value) for value in cf_mat_flat]

    group_percentages = ["{0:.2%}".format(value) for value in 
                         cf_mat_flat/np.sum(cf_mat_flat)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in 
              zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    if ax_cnt is None:        
        plt.figure()
        
    ax = sns.heatmap(cf_matrix, ax=axs, annot=labels, fmt='', annot_kws={"size": 14}, cmap='Blues')

    #ax.set_title('CM: 0-pixel tolerance\n');    
    ax.set_title(f'{title}\n Images={img_cnt}');
    ax.set_xlabel('\nPredicted Centers')
    ax.set_ylabel('Actual Centers');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['True','False'])
    ax.yaxis.set_ticklabels(['True','False'])

    if is_save:
        if ax_cnt is None:
            # save each plot
            plt.savefig(fullpath, dpi=100, bbox_inches='tight')
        else:
            # save the last plot
            if ax_cnt==2:
                plt.savefig(fullpath, dpi=100, bbox_inches='tight')
    
            
        