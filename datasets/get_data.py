import os
import pandas as pd
from typing import Literal, Optional
from .csainit import csa_initialization
from .teinit import te_initialization
from .headdict import get_score_head
from .dataset import CLIPDataset


def getdata(task:Literal['CSA', 'TE'], site:Literal['Wrist','MCP','Foot'], feature:Literal['TSY','SYN','BME'], 
            filt:Optional[list]=None, score_sum:bool=False):
    path_default = {'CSA':r'./datasets/csa_init.csv', 'TE':r'./datasets/te_init.csv'}
    paths = path_default[task]

    if not os.path.exists(paths):
        df:pd.DataFrame = csa_initialization() if task=='CSA' else te_initialization()
        df.to_csv(paths)
    else:
        df:pd.DataFrame = pd.read_csv(paths)
    path_column = [col for col in df.columns if f'{site}' in col]
    pre = ['ID', 'DATE', 'ID_DATE']
    pre.extend(path_column)
    score_column = get_score_head(site, feature)
    target:pd.DataFrame = df[pre]    # ID, DATE, ID_DATE, SITE_TRA, SITE_COR   
    # 筛除某些部分
    if filt: target = target[target['ID'].isin(filt)]

    return CLIPDataset(target, path_column, score_column, score_sum, path_flag=True), target.shape[0]

