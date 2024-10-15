from typing import Literal, Optional
import pandas as pd
import torch
import platform
from tqdm import tqdm
from .cam_components import CAMAgent
from .datasets.get_data import getdata
from .models.get_model import getmodel
from .trained_weights.get_weight import getweight
from .utils.get_head import return_head, return_head_gt
from .utils.get_path import getpath
from torch.utils.data import DataLoader

def cam_main_process(task:Literal['CSA', 'TE'], site:Literal['Wrist','MCP','Foot'],
                 feature:Literal['TSY','SYN','BME'], order:int=0, 
                 score_sum:bool=False, filt:Optional[list]=None):
    
    # 模型本身和权重都需要site feature
    model = getmodel(site, feature, score_sum)  # DONE!
    model = getweight(model, site, feature, order)  # DONE!
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    target_layer = [model.features[-1]]
    # 数据读取需要task, site, feature
    # 数据返回应该是 img, scores, path (id, date)
    data, _ = getdata(task, site, feature, filt, score_sum)
    # filt 用来控制哪些id会被使用
    # data: x,y,z: img, scores, path

    # CAM生成不需要采用head的模式，只需要正常往后推就可以
    # -------------------------------------------------- initialize camagent ----------------------------------------------- #
    Agent = CAMAgent(model, target_layer, data,  
            groups=1, ram=True,
            # optional:
            cam_method='fullcam', name_str=f'ramris_site{site}_fea{feature}',# cam method and im paths and cam output
            batch_size=1, select_category=0,  # info of the running process
            rescale='norm',  remove_minus_flag=False, scale_ratio=1.5,
            feature_selection='all', feature_selection_ratio=0.05,  # feature selection
            randomization=None,  # model randomization for sanity check
            use_pred=False,
            rescaler=None,  # outer scaler
            cam_type='3D'  # output 2D or 3D
            )

    # -------------------------------------------------------- record ----------------------------------------------------- #
    valdata = DataLoader(dataset=data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Agent.creator_main(valdata, 'Default', False, True, None, False, None)
    # for x, y, z in data:
    #     x = x.to(device) # [batch, 1, L, W]
    #     y = y.to(device) # [label/scores float]
    #     # z path/number of the CSA/TE [int]
    #     cam = Agent.indiv_return(x, 1, None, False)
    #     # [batch, 1(Group), 1(category in list), D, L, W]


if __name__=='__main__':
    for site in ['Wrist', 'MCP', 'Foot']:
        for feature in ['TSY','SYN','BME']:
            cam_main_process('CSA', site, feature)