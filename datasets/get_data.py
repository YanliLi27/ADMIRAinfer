import os
import pandas as pd
from typing import Literal, Optional
from .csainit import csa_initialization
from .teinit import te_initialization
from .allinit import all_initialization
from .headdict import get_score_head
from .dataset import CLIPDataset3D


def getdata(task:Literal['CSA', 'TE'], site:Literal['Wrist','MCP','Foot'], feature:Literal['TSY','SYN','BME'], view:list[str]=['TRA', 'COR'], 
            filt:Optional[list]=None, score_sum:bool=False, path_flag:bool=True):
    path_default = {'CSA':r'./datasets/csa_init.csv', 'TE':r'./datasets/te_init.csv'}
    paths = path_default[task]

    if not os.path.exists(paths):
        df:pd.DataFrame = csa_initialization() if task=='CSA' else te_initialization()
        df.to_csv(paths)
    else:
        df:pd.DataFrame = pd.read_csv(paths)
    if len(view)>1:
        path_column = [col for col in df.columns if f'{site}_' in col]
    else:
        path_column = [col for col in df.columns if f'{site}_{view[0]}' in col]
    pre = ['ID', 'DATE', 'ID_DATE']
    pre.extend(path_column)
    score_column = get_score_head(site, feature)
    pre.extend(score_column)
    target:pd.DataFrame = df[pre]    # ID, DA.dropna()TE, ID_DATE, SITE_TRA, SITE_COR   
    # 筛除某些部分
    if filt: target = target[target['ID'].isin(filt)]
    target = target.dropna()

    return CLIPDataset3D(target, path_column, score_column, score_sum, path_flag), target.shape[0]


def getdata_ult(task:Literal['CSA', 'TE', 'ALL'], site:Literal['Wrist','MCP','Foot'], feature:Literal['TSY','SYN','BME'], view:list[str]=['TRA', 'COR'], 
            filt:Optional[list]=None, score_sum:bool=False, order:int=0, path_flag:bool=True):
    if task in ['CSA', 'TE']: 
        assert task!='ALL'
        return getdata(task, site, feature, view, filt, score_sum, path_flag)

    path_default = {'ALL':f'./datasets/all/all_{site}_{feature}_{task}_sum{score_sum}_{order}.csv'}
    paths = path_default[task]
    if not os.path.exists(os.path.dirname(paths)): os.makedirs(os.path.dirname(paths))
    if not os.path.exists(paths):
        df:pd.DataFrame = all_initialization(site=site, feature=feature, order=order, sum_score=score_sum)
        df.to_csv(paths)
    else:
        df:pd.DataFrame = pd.read_csv(paths)
    if len(view)>1:
        path_column = [col for col in df.columns if f'{site}_' in col]   # Wrist_TRA/COR
    else:
        path_column = [col for col in df.columns if f'{site}_{view[0]}' in col]  # Wrist_view[0]
    pre = ['ID', 'DATE', 'ID_DATE']
    pre.extend(path_column)
    score_column = get_score_head(site, feature)
    pre.extend(score_column)
    target:pd.DataFrame = df[pre]    # ID, DA.dropna()TE, ID_DATE, SITE_TRA, SITE_COR   
    # 筛除某些部分
    if filt: target = target[target['ID'].isin(filt)]
    target = target.dropna()

    return CLIPDataset3D(target, path_column, score_column, score_sum, path_flag), target.shape[0]

