from seg_components import model_seg_find, data_process, segcor, segtra
from typing import Literal, Union, Optional, List
from datasets.get_data import getdata
from models.get_model import getmodel
from trained_weights.get_weight import getweight
from cam_components import CAMAgent
from utils.log import Record
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader


def main_process3d(task:Literal['CSA', 'TE'], site:Literal['Wrist','MCP','Foot'],
                    feature:Literal['TSY','SYN','BME'], view:Optional[List[str]]=['TRA', 'COR'], 
                    score_sum:bool=False, filt:Optional[list]=None):
    if not view:
        view = ['COR'] if feature in ['SYN', 'BME'] else ['TRA']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get segmentation model
    model_seg_tra = model_seg_find(in_channel=1, n_classes=33, width=1, site='tra', fold_order=0)
    model_seg_cor = model_seg_find(in_channel=3, n_classes=19, width=1, site='cor', fold_order=0)
    # get regression model
    model = getmodel(site, feature, view, score_sum)  # DONE!
    model = getweight(model, site, feature, view, order=0)  # DONE!
    model = model.to(device)
    target_layer = [model.features[-1]]
    # get data
    data, _ = getdata(task, site, feature, view, filt, score_sum, path_flag=False)
    data = DataLoader(dataset=data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # -------------------------------------------------- segmentation model ----------------------------------------------- #
    model_seg_tra = model_seg_find(in_channel=1, n_classes=33, width=1, site='tra', fold_order=0)
    model_seg_cor = model_seg_find(in_channel=3, n_classes=19, width=1, site='cor', fold_order=0)
    model_seg_tra = model_seg_tra.to(device)
    model_seg_cor = model_seg_cor.to(device)
    # -------------------------------------------------- initialize camagent ----------------------------------------------- #
    Agent = CAMAgent(model, target_layer, data,  
            groups=len(view), ram=True,
            # optional:
            cam_method='fullcam', name_str=f'ramris_site{site}_fea{feature}_multiview',# cam method and im paths and cam output
            batch_size=1, select_category=0,  # info of the running process
            rescale='norm',  remove_minus_flag=False, scale_ratio=1.5,
            feature_selection='all', feature_selection_ratio=0.05,  # feature selection
            randomization=None,  # model randomization for sanity check
            use_pred=False,
            rescaler=None,  # outer scaler
            cam_type='3D'  # output 2D or 3D
            )
    # -------------------------------------------------------- record ----------------------------------------------------- #
    rec = Record()
    general = Record()
    tra_cm_general = Record()
    cor_cm_general = Record()
    # [N*[TP, FP, TN, FN], ...]
    sum_tra = []
    sum_cor = []
    freq_tra = np.zeros((33), dtype=int)
    freq_cor = np.zeros((19), dtype=int)
    # -------------------------------------------------------- record ----------------------------------------------------- #

    valdata, _ = getdata(task, site, feature, view, filt, score_sum, path_flag=False)
    valdataloader = DataLoader(dataset=valdata, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -------------------------------------------------------- get results ----------------------------------------------------- #
    # calculate the mean activations of each regions -- tendon extended to TSY, SYN extended to remaining tissue, BME inside the bones
    counter = 0
    for x, y in tqdm(valdataloader):
        # just get the segmentation of all categories and merge them later
        x, tra_x, cor_x = data_process(x, num_input=len(view))
        x = x.to(device) # [batch, channel, D, L, W]
        # y = y.to(device) # [batch, label/scores float]
        # z path/number of the CSA/TE [batch, int]
        cam = Agent.indiv_return(x, 1, None, False)
        # [B(batch), 1/2(Group), 1(category in list), D, L, W]

        # segmentation -- can be done anyway 
        tra_y = segtra(tra_x, model_seg_tra, merge=False) if 'TRA' in view else None
        cor_y = segcor(cor_x, model_seg_cor, merge=False) if 'COR' in view else None
        # [B, 33, D, L, W], [B, 19, D, L, W]

        # Two situation: TRA/COR or TRA&COR
        if len(view)>1: tra_cam, cor_cam = np.expand_dims(cam[:, 0, 0, :], axis=1), np.expand_dims(cam[:, 1, 0, :], axis=1) 
            # [B, 1, D, L, W], [B, 1, D, L, W]
        else: tra_cam = cor_cam = np.expand_dims(cam[:, 0, 0, :], axis=1)
            # [B, 1, D, L, W]
        
        tra_con = tra_cam * tra_y if tra_y else None
        cor_con = cor_cam * cor_y if cor_y else None
        # [B, 1, D, L, W] * [B, 33/19, D, L, W] --> [B, 33/19, D, L, W]

        # record the frequency of the anatomical regions
        for i in range(x.shape[0]):  # [B, C, D, L, W]
            if tra_y:
                segfreqtra = np.sum(tra_y[i], axis=(1, 2, 3))  # [33]
                mask = segfreqtra>0
                segfreqtra[mask] = 1
                freq_tra+= segfreqtra.astype(int)
            if cor_y:
                segfreqcor = np.sum(cor_y[i], axis=(1, 2, 3))  # [19]
                mask = segfreqcor>0
                segfreqcor[mask] = 1
                freq_cor+= segfreqcor.astype(int)
        freq_tra = np.asarray(freq_tra) if tra_y else None
        freq_cor = np.asarray(freq_cor) if cor_y else None

        tra = np.squeeze(np.sum(tra_con, axis=(2, 3, 4))/(np.sum(tra_y, axis=(2, 3, 4))+1e-7)) if tra_y else None # [B, 33]
        cor = np.squeeze(np.sum(cor_con, axis=(2, 3, 4))/(np.sum(cor_y, axis=(2, 3, 4))+1e-7)) if cor_y else None # [B, 19]

        if tra: sum_tra.extend(tra)  # [b+b+b+..., 33]
        if cor: sum_cor.extend(cor)  # [b+b+b+..., 19]
        rec(TRA=tra,COR=cor)
        if counter%100==0:
            rec.summary(save_path='./output/record3d.csv')

    num_cm = []
    if len(sum_tra)>0:
        mean_tra = np.mean(np.asarray(sum_tra), axis=0)  # [N, 33] --> [33]
        std_tra = np.std(np.asarray(sum_tra), axis=0)
    else:
        mean_tra = std_tra = None

    if len(sum_cor)>0:
        mean_cor = np.mean(np.asarray(sum_cor), axis=0)  # [N, 19] --> [19]
        std_cor = np.std(np.asarray(sum_cor), axis=0)
    else: mean_cor = sum_cor = None

    
    general(sum_tra=sum_tra, sum_cor=sum_cor, num_cm=num_cm,
            mean_tra=mean_tra, std_tra=std_tra, mean_cor=mean_cor, std_cor=std_cor
            )
    rec.summary(save_path='./output/record3d.csv')
    general.summary(save_path='./output/sum3d.csv')

    # [33/19] save separately
    if mean_tra:
        tra_cm_general(background=mean_tra[0] , skin=mean_tra[1], vessel=mean_tra[2],
                    tissue=mean_tra[3], FCU=mean_tra[4], FPL=mean_tra[5],
                    FCR=mean_tra[6], PL=mean_tra[7], FDS=mean_tra[8],
                    FDP=mean_tra[9], EPL=mean_tra[10], ED=mean_tra[11],
                    EDM=mean_tra[12], ECU=mean_tra[13], EPB=mean_tra[14],
                    APL=mean_tra[15], ECRB=mean_tra[16], ECRL=mean_tra[17],
                    Scaphoid=mean_tra[18], Lunate=mean_tra[19], Triquetrum=mean_tra[20],
                    Pisiform=mean_tra[21], Trapezium=mean_tra[22], Trapezoid=mean_tra[23],
                    Capitate=mean_tra[24], Hamate=mean_tra[25], Ulna=mean_tra[26],
                    Radius=mean_tra[27], Metacarpal1=mean_tra[28], Metacarpal2=mean_tra[29],
                    Metacarpal3=mean_tra[30], Metacarpal4=mean_tra[31], Metacarpal5=mean_tra[32],
                    background_std=std_tra[0] , skin_std=std_tra[1], vessel_std=std_tra[2],
                    tissue_std=std_tra[3], FCU_std=std_tra[4], FPL_std=std_tra[5],
                    FCR_std=std_tra[6], PL_std=std_tra[7], FDS_std=std_tra[8],
                    FDP_std=std_tra[9], EPL_std=std_tra[10], ED_std=std_tra[11],
                    EDM_std=std_tra[12], ECU_std=std_tra[13], EPB_std=std_tra[14],
                    APL_std=std_tra[15], ECRB_std=std_tra[16], ECRL_std=std_tra[17],
                    Scaphoid_std=std_tra[18], Lunate_std=std_tra[19], Triquetrum_std=std_tra[20],
                    Pisiform_std=std_tra[21], Trapezium_std=std_tra[22], Trapezoid_std=std_tra[23],
                    Capitate_std=std_tra[24], Hamate_std=std_tra[25], Ulna_std=std_tra[26],
                    Radius_std=std_tra[27], Metacarpal1_std=std_tra[28], Metacarpal2_std=std_tra[29],
                    Metacarpal3_std=std_tra[30], Metacarpal4_std=std_tra[31], Metacarpal5_std=std_tra[32],
                    )
        tra_cm_general.summary(save_path='./output/record_tra3d.csv')
    if mean_cor:
        cor_cm_general(background=mean_cor[0], skin=mean_cor[1], vessel=mean_cor[2],
                    tissue=mean_cor[3],
                    Scaphoid=mean_cor[4], Lunate=mean_cor[5], Triquetrum=mean_cor[6],
                    Pisiform=mean_cor[7], Trapezium=mean_cor[8], Trapezoid=mean_cor[9],
                    Capitate=mean_cor[10], Hamate=mean_cor[11], Ulna=mean_cor[12],
                    Radius=mean_cor[13], Metacarpal1=mean_cor[14], Metacarpal2=mean_cor[15],
                    Metacarpal3=mean_cor[16], Metacarpal4=mean_cor[17], Metacarpal5=mean_cor[18],
                    background_std=std_cor[0], skin_std=std_cor[1], vessel_std=std_cor[2],
                    tissue_std=std_cor[3],
                    Scaphoid_std=std_cor[4], Lunate_std=std_cor[5], Triquetrum_std=std_cor[6],
                    Pisiform_std=std_cor[7], Trapezium_std=std_cor[8], Trapezoid_std=std_cor[9],
                    Capitate_std=std_cor[10], Hamate_std=std_cor[11], Ulna_std=std_cor[12],
                    Radius_std=std_cor[13], Metacarpal1_std=std_cor[14], Metacarpal2_std=std_cor[15],
                    Metacarpal3_std=std_cor[16], Metacarpal4_std=std_cor[17], Metacarpal5_std=std_cor[18])
        cor_cm_general.summary(save_path='./output/record_cor3d.csv')






if __name__=='__main__':
    for site in ['Wrist']: #, 'MCP', 'Foot']:
        for feature in ['TSY','SYN','BME']:
            main_process3d('CSA', site, feature, ['TRA', 'COR'], True)