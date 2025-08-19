# for cam visualization


from typing import Literal, Optional, List
import pandas as pd
import torch
import os
import numpy as np
import platform
from tqdm import tqdm
import SimpleITK as sitk
from cam_components import CAMAgent
from datasets.get_data import getdata, getdata_eac
from models.get_model import getmodel
from trained_weights.get_weight import getweight
from utils.get_head import return_head, return_head_gt
from torch.utils.data import DataLoader

def cam_2view_main_process(task:Literal['CSA', 'TE', 'EAC'], site:Literal['Wrist','MCP','Foot'],
                           feature:Literal['TSY','SYN','BME'], view:Optional[List[str]]=['TRA', 'COR'], 
                           score_sum:bool=False, filt:Optional[list]=None):
    if not view:
        view = ['COR'] if feature in ['SYN', 'BME'] else ['TRA']
    # 模型本身和权重都需要site feature
    model = getmodel(site, feature, view, score_sum)  # DONE!
    model = getweight(model, site, feature, score_sum, view, order=0)  # DONE!
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    target_layer = [model.features[-1]]
    # 数据读取需要task, site, feature
    # 数据返回应该是 img, scores, path (id, date)
    csa_filt = [636, 25, 154, 162, 66, 165, 274, 528, 712, 122, 204, 473, 501, 313, 497, 
            87, 534, 835, 607, 627, 375, 98, 225, 185, 310, 484, 795, 101, 814, 489, 
            382, 213, 16, 102, 426, 399, 765, 46, 787, 112, 257, 737, 329, 1, 770, 510, 
            674, 455, 686, 421, 628, 307, 367, 161, 447, 265, 145, 251, 656, 569, 600, 
            696, 52, 500, 85, 440, 247, 202, 322, 766, 679, 142, 478, 217, 429, 796, 
            401, 622, 448, 206, 270, 519, 550, 326, 5, 108, 403, 116, 332, 311, 243, 
            333, 248, 585, 42, 275, 513, 526, 292, 602, 129, 188, 197, 130, 476, 71, 
            176, 335, 131, 291, 598, 135, 431, 560, 722, 133, 314, 781, 156, 356]
    
    # updated
    eac_filt = [3599, 3914, 3893, 4124, 4148, 3679, 3436, 3454, 3945, 3957]
    filt = ['Csa' + str(x).zfill(3) for x in csa_filt] if task=='CSA' else ['Arth' + str(x).zfill(4) for x in eac_filt]
    data, _ = getdata(task, site, feature, view, filt, score_sum, path_flag=False) \
        if task=='CSA'  else getdata_eac(task, site, feature, view, None, score_sum, path_flag=False)
    data = DataLoader(dataset=data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    # filt 用来控制哪些id会被使用
    # data: x,y,z: img, scores, path
    namestr = f'ramris{task}_site{site}_fea{feature}_{view[0]}' if len(view)<2 else f'ramris{task}_site{site}_fea{feature}_multiview'
    # CAM生成不需要采用head的模式，只需要正常往后推就可以
    # -------------------------------------------------- initialize camagent ----------------------------------------------- #
    Agent = CAMAgent(model, target_layer, data,  
            groups=len(view), ram=True,
            # optional:
            cam_method='fullcam', name_str=namestr,# cam method and im paths and cam output
            batch_size=1, select_category=0,  # info of the running process
            rescale='norm',  remove_minus_flag=False, scale_ratio=1.5,
            feature_selection='all', feature_selection_ratio=0.05,  # feature selection
            randomization=None,  # model randomization for sanity check
            use_pred=False,
            rescaler=None,  # outer scaler
            cam_type='3D'  # output 2D or 3D
            )

    # -------------------------------------------------------- record ----------------------------------------------------- #
    valdata, _ = getdata(task, site, feature, view, filt, score_sum, path_flag=True)\
        if task=='CSA'  else getdata_eac(task, site, feature, view, filt, score_sum, path_flag=True)
    valdataloader = DataLoader(dataset=valdata, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Agent.creator_main(valdataloader, 'Default', False, True, None, False, None) # 无法提供路径
    cnt = 0
    for x, y, z in tqdm(valdataloader):
        x = x.to(device) # [batch, channel, D, L, W]
        # y = y.to(device) # [batch, label/scores float]
        # z path/number of the CSA/TE [batch, int]
        cam = Agent.indiv_return(x, 1, None, False)
        # [batch, 2(Group), 1(category in list), D, L, W
        for b in range(cam.shape[0]):
            cnt+=1
            for g in range(cam.shape[1]):
                save_name = f'./output/{namestr}/cam/fullcam/scalenorm_rmFalse_feall0.05/cate0/ID{z[b]}_view{view[g]}_sc{y[b].numpy()[0]}.nii.gz'
                # D:\ESMIRAcode\RAMRISinfer\output\ramris_siteWrist_feaTSY_TRA\cam\fullcam\scalenorm_rmFalse_feall0.05\cate0
                writter = sitk.ImageFileWriter()
                writter.SetFileName(save_name)

                savecam = cam[b][g][0]
                writter.Execute(sitk.GetImageFromArray(savecam))

                origin_save_name = f'./output/{namestr}/cam/fullcam/scalenorm_rmFalse_feall0.05/cate0/ID{z[b]}_view{view[g]}_origin.nii.gz'
                if not os.path.isfile(origin_save_name):
                    writter.SetFileName(origin_save_name)
                    # [batch, organ_groups, z, y, x, channel] to [batch, organ_groups, z, y, x]
                    writter.Execute(sitk.GetImageFromArray(x.cpu().numpy()[b][g]))



if __name__=='__main__':
    for site in ['Wrist']: #, 'MCP', 'Foot']:
        for feature in ['TSY','SYN','BME']:
            for view in [['TRA'], ['COR']]:
                cam_2view_main_process('EAC', site, feature, view, True)