import os.path as osp
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
from scipy.ndimage import binary_dilation
import seaborn as sns
import matplotlib.pyplot as plt

from utils import makeDirectory
#from visualization import save_confusion_matrix

def perform_objective_evaluation(df_all:pd.DataFrame, io_paths:dict):
    
    tags = ['train', 'valid', 'test', 'org']

    LOD = []
    for tag in tags:
        if tag=='org':
            df_name = df_all[(df_all['origin']=='org')].reset_index(drop=True).copy(deep=True)
        else:
            df_name = df_all[ (df_all['data']==tag) & (df_all['origin']=='org')].reset_index(drop=True).copy(deep=True)
        
        # perform evaluation
        all_eval_dict = save_objective_results(df_name=df_name, paths=io_paths, data_tag=tag, is_together=True)
        
        # cleaning memory
        del df_name
        
        # converting and appending to create dataframe for plots
        for tol, eval_dict in all_eval_dict.items():
            info_dict={}
            info_dict['data'] = tag
            info_dict['tol'] = tol.split('_')[1]
            for met, scr in eval_dict.items():
                info_dict[f'{met}'] = scr
            
            LOD.append(info_dict)

    df_eval = pd.DataFrame(LOD)
    return df_eval

def compute_performance_numbers(perf_dict:dict):
    TP = perf_dict['TP']
    FP = perf_dict['FP']
    FN = perf_dict['FN']
    TN = perf_dict['TN']
       
    # Precision:
    prec = TP/(TP+FP)
    
    # Recall
    rec = TP/(TP+FN)
    
    # F-1 score
    f1_scr = (2 * prec * rec) / (prec + rec)
    
    eval_dict = {'Precision': float("{0:0.2f}".format(prec)), 
                 'Recall': float("{0:0.2f}".format(rec)),
                 'F1_score': float("{0:0.2f}".format(f1_scr))}
    
    return eval_dict
    
def get_cent_mask_from_yolo_labels(bb_path:str, im_width:int, im_height:int):
    
    # initialize image RGBA
    msk_pred = np.zeros([im_height, im_width, 4],dtype=np.uint8)    
       
    # open and read yolo format bounding box annotations 
    file_obj = open(bb_path, 'r')
    lines = file_obj.read().splitlines()
    for line in lines:
        yolo_format = line.split(' ')
        # retrive box centers  
        x_cent = int(float(yolo_format[1]) * im_width)
        y_cent = int(float(yolo_format[2]) * im_height)        
        
        # Set mask red- and aplha-channel as prediction info 
        msk_pred[y_cent, x_cent, 0] = 255 
        msk_pred[y_cent, x_cent, 3] = 255                        
    
    return msk_pred


def get_confusion_mat_numbers(gt:np.array, pred:np.array, tol:int=0):    
        
    # check in tolrance is specified
    perf_dict = {}        
    
    if tol:        
        sz = 2*tol+1
        se = np.ones([sz, sz], dtype=np.bool_)
        dgm = binary_dilation(gt, se)
        dpm = binary_dilation(pred, se)
        
        perf_dict['TP'] = np.sum(np.logical_and(pred==True, dgm==True))        
        perf_dict['FP'] = np.sum(np.logical_and(pred==True, dgm==False))
        perf_dict['FN'] = np.sum(np.logical_and(dpm==False, gt==True))               
        #perf_dict['TN'] = np.sum(np.logical_and(dpm==False, dgm==False))
        perf_dict['TN'] = (gt.shape[0]*gt.shape[1])-(perf_dict['TP']+perf_dict['FP']+perf_dict['FN'])
    else:
        perf_dict['TP'] = np.sum(np.logical_and(pred==True, gt==True))    
        perf_dict['FP'] = np.sum(np.logical_and(pred==True, gt==False))
        perf_dict['FN'] = np.sum(np.logical_and(pred==False, gt==True))
        perf_dict['TN'] = np.sum(np.logical_and(pred==False, gt==False))
    
    return perf_dict



def save_objective_results(df_name:pd.DataFrame, paths:dict, data_tag:str, is_together:bool=False):
    
    op_path = osp.join(paths['perf_plots'], data_tag)
    makeDirectory(op_path)

    perf_P0_tol_dict = {'TP':0, 'FN':0, 'FP':0, 'TN':0}
    perf_P1_tol_dict = {'TP':0, 'FN':0, 'FP':0, 'TN':0}
    perf_P2_tol_dict = {'TP':0, 'FN':0, 'FP':0, 'TN':0}

    tot_img = len(df_name.index)
    for idx in tqdm.tqdm(range(tot_img)):

        im_name = df_name.at[idx, 'name']
        img_fullpath = df_name.at[idx, 'image_fullpath']
        msk_fullpath = df_name.at[idx, 'mask_fullpath']

        # read original image and mask
        img = np.array(Image.open(img_fullpath))
        msk_gt = np.array(Image.open(msk_fullpath))    
        msk_gt_b = msk_gt[:,:,1].astype(np.bool_)

        # read predicted labels
        #label_key = f'pred_labels_{data_tag}'
        label_key = 'pred_labels_org' # Hard coding for now (can be customized)
        labels_fullpath = osp.join(paths[label_key],im_name.replace('tif', 'txt'))
        im_h, im_w = img.shape[0], img.shape[1]

        # generate predicted mask (centre locations)
        msk_pred = get_cent_mask_from_yolo_labels(bb_path=labels_fullpath, im_width=im_w, im_height=im_h)
        msk_pred_b = msk_pred[:,:,0].astype(np.bool_)

        # performance -- pixel accurate 
        perf_0px_tol = get_confusion_mat_numbers(gt=msk_gt_b, pred=msk_pred_b, tol=0)    
        #plot_confusion_matrix(perf_dict=perf_0px_tol, title='CM: 0-Pixel Tolerance')
        for key, val in perf_P0_tol_dict.items():
            perf_P0_tol_dict[key]=val+perf_0px_tol[key]

        # performance -- 1-pixel tolerance
        perf_1px_tol = get_confusion_mat_numbers(gt=msk_gt_b, pred=msk_pred_b, tol=1)    
        for key, val in perf_P1_tol_dict.items():
            perf_P1_tol_dict[key]=val+perf_1px_tol[key]

        #plot_confusion_matrix(perf_dict=perf_1px_tol, title='CM: 1-Pixel Tolerance')

        # performance -- 1-pixel tolerance
        perf_2px_tol = get_confusion_mat_numbers(gt=msk_gt_b, pred=msk_pred_b, tol=2)    
        #plot_confusion_matrix(perf_dict=perf_2px_tol, title='CM: 2-Pixel Tolerance')
        for key, val in perf_P2_tol_dict.items():
            perf_P2_tol_dict[key]=val+perf_2px_tol[key]

    # Overall performance
    all_eval_dict = {}
    all_eval_dict['eval_p0_tol'] = compute_performance_numbers(perf_dict=perf_P0_tol_dict)
    all_eval_dict['eval_p1_tol'] = compute_performance_numbers(perf_dict=perf_P1_tol_dict)
    all_eval_dict['eval_p2_tol'] = compute_performance_numbers(perf_dict=perf_P2_tol_dict)
    
    # confusion matrix plots        
    if is_together:
        # plot all in a single plot
        #fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[4,1,0.2]))
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(22, 4), sharex=True)
           
        save_confusion_matrix(axs=axs[0], perf_dict=perf_P0_tol_dict, title='CM: 0-Pixel Tolerance', 
                            img_cnt=tot_img, fullpath='', ax_cnt=0)

        save_confusion_matrix(axs=axs[1], perf_dict=perf_P1_tol_dict, title='CM: 1-Pixel Tolerance', 
                            img_cnt=tot_img, fullpath='', ax_cnt=1)

        fullpath = osp.join(op_path,f'joint_conf_mat_plot.png')
        save_confusion_matrix(axs=axs[2], perf_dict=perf_P2_tol_dict, title='CM: 2-Pixel Tolerance', 
                            img_cnt=tot_img, fullpath=fullpath, ax_cnt=2)
        
        print(f'Confusion matrix plots saved at:\n {op_path}')
        
    else:
        save_confusion_matrix(perf_dict=perf_P0_tol_dict, title='CM: 0-Pixel Tolerance', 
                            img_cnt=tot_img, fullpath=osp.join(op_path,f'cm_pixel_tol_0.png'))

        save_confusion_matrix(perf_dict=perf_P1_tol_dict, title='CM: 1-Pixel Tolerance', 
                            img_cnt=tot_img, fullpath=osp.join(op_path,f'cm_pixel_tol_1.png'))

        save_confusion_matrix(perf_dict=perf_P2_tol_dict, title='CM: 2-Pixel Tolerance', 
                            img_cnt=tot_img, fullpath=osp.join(op_path,f'cm_pixel_tol_2.png'))
        
        print(f'\n Confusion matrix plots saved at:\n {op_path}')
    #end-if
    
    return all_eval_dict

def save_confusion_matrix(perf_dict:dict, title:str, fullpath:str='', img_cnt:int=1, axs=None, ax_cnt=None):
    _plot_confusion_matrix(perf_dict=perf_dict, title=title, img_cnt=img_cnt, is_save=True,
                          fullpath=fullpath, axs=axs, ax_cnt=ax_cnt)
 
        
def _plot_confusion_matrix(perf_dict:dict, title:str, img_cnt:int=1, is_save:bool=False, fullpath:str='', axs=None, ax_cnt=None):
        
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
