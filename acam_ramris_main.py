from seg_components import model_seg_find
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
    # get segmentation model
    model_seg_tra = model_seg_find(in_channel=1, n_classes=33, width=1, site='tra', fold_order=0)
    model_seg_cor = model_seg_find(in_channel=3, n_classes=19, width=1, site='cor', fold_order=0)
    # get regression model
    model = getmodel(site, feature, view, score_sum)  # DONE!
    model = getweight(model, site, feature, view, order=0)  # DONE!
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    target_layer = [model.features[-1]]
    # get data
    data, _ = getdata(task, site, feature, view, filt, score_sum, path_flag=False)
    data = DataLoader(dataset=data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

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
    sum_cm_tra = []
    std_cm_tra = []
    sum_cm_cor = []
    std_cm_cor = []
    cm_flag = 0
    freq_tra = np.zeros((33), dtype=int)
    freq_cor = np.zeros((19), dtype=int)
    # -------------------------------------------------------- record ----------------------------------------------------- #

    valdata, _ = getdata(task, site, feature, view, filt, score_sum, path_flag=False)
    valdataloader = DataLoader(dataset=valdata, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -------------------------------------------------------- get results ----------------------------------------------------- #
    cnt = 0
    for x, y, z in tqdm(valdataloader):
        x = x.to(device) # [batch, channel, D, L, W]
        # y = y.to(device) # [batch, label/scores float]
        # z path/number of the CSA/TE [batch, int]
        cam = Agent.indiv_return(x, 1, None, False)
        # [batch, 1/2(Group), 1(category in list), D, L, W]




if __name__=='__main__':
    for site in ['Wrist']: #, 'MCP', 'Foot']:
        for feature in ['TSY','SYN','BME']:
            main_process3d('CSA', site, feature, ['TRA', 'COR'], True)