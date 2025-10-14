from typing import Literal, Optional, List
import pandas as pd
from functools import reduce
import os
import torch
from datasets.get_data import getdata_ult
from models.get_model import getmodel
from trained_weights.get_weight import getweight_outside
from utils.get_head import return_head, return_head_gt
from torch.utils.data import DataLoader
from tqdm import tqdm

from otherutils.label_id_function import label_id

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# 从 E:\ADMIRA_models\weights\sumFalse、True 获取 {BIO}__{SITE}_2dirc_fold{FOLD}.model的权重 BME__Foot_2dirc_fold3.model

# 从 getdata获得对应的fold的 monitoring 数据
# 从 E:\ADMIRA_models\split\sumFalse、True 获取monitoring的数据分割：
# 用pkl_reader把它们转化为csv数据，然后用和此处的dataset相同的形式进行保存
# 【un22_EAC_CSA_ATL__{SITE}_2dirc_1.pkl】  
# 里面存的是 [5split  *[id[path1:cs, path2:cs, path3:cs, ...]的list
# 【un22_EAC_CSA_ATL__{SITE}_{BIO}_2reader_1__sc.pkl 
# 里面存的是对应BIO的 [5split  *[id[site1_array, site2_array], id[[site1_array, site2_array]]

# 每次把对应的数据和对应的模型进行一波inference，然后存到一个csv当中，首先进行一个site下的全split合并
# 然后合并三个site到一个csv，最后通过桌面的hold的标记增加一列标注他们的类别
# 然后根据桌面的CAM.xlsx对之前的做了obeserver study的进行增列标注

# 同样的用训练好的模型对TE进行一个inference - 使用main.py，保存成类似的情况下，然后类似合并


def main_process(task:Literal['TE', 'CSA', 'ALL'], site:Literal['Wrist', 'MCP', 'Foot'],
                 feature:Literal['TSY','SYN','BME'], 
                 view:List[str]=['TRA', 'COR'],
                 order:int=0, score_sum:bool=False, filt:Optional[list]=None,
                 name_str:str='temp'):
    model = getmodel(site, feature, view, score_sum)  # DONE!
    model = getweight_outside(model, r'E:\ADMIRA_models\weights', site, feature, score_sum, view, order)  # DONE!
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    # obtain data
    # 数据读取需要task, site, feature
    # 数据返回应该是 img, scores, path (id, date)
    data, maxidx = getdata_ult(task, site, feature, view, filt, score_sum, order, True)
    # CSA那个需要返回所有选择的CSA的列表并进行On-fly选中间的切片
    # TE那个需要返回所有的Timepoint的列表并进行On-fly选中间的切片
    # x,y,z: img, scores, path
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1)

    # create results dict for csv file
    res_head= ['ID', 'ScanDatum', 'ID_Timepoint']
    if not score_sum:
        res_head.extend(return_head(site, feature))
        res_head.extend(return_head_gt(site, feature))
    else:
        res_head.extend([f'sums', f'sums'])
    df = pd.DataFrame(index=range(maxidx), columns=res_head)
    idx = 0
    for x, y, z in tqdm(dataloader):
        x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # Tensor
        with torch.no_grad():
            pred:torch.Tensor = model(x)  # [B, num_scores] Tensor
            for i in range(x.shape[0]):
                pid, ptp = z[i].split('_')  # getpath Done!
                row = [pid, ptp, f'{pid}_{ptp}']
                row.extend(pred[i].cpu().numpy())
                row.extend(y[i].cpu().numpy())
                df.loc[idx] = row
                idx += 1
        # 用pd.concat([df, new_row], ignore_index=True)来添加新的一行数据
    path = f'./output/all_te_{name_str}/0unmerged_{site}_{feature}_{task}_sum{score_sum}_{order}.csv'
    dir = os.path.dirname(path)
    if not os.path.exists(dir): os.makedirs(dir)
    df.to_csv(path)
    return df
    # inference:
    # 直接for x,y,z in Dataloader():
    # pred = model(x)
    # 根据z划分出ID和time_point(只需要有时间就可以了，
    # 再存一个ID_TP: CSA001_20140901用来合并
    # 存到一个csv里面

def merge_fold_process(task:Literal['TE', 'CSA', 'ALL'], site:Literal['Wrist', 'MCP', 'Foot'],
                 feature:Literal['TSY','SYN','BME'], 
                 view:List[str]=['TRA', 'COR'],
                 score_sum:bool=False, filt:Optional[list]=None,
                 name_str:str='temp'):
    df_list:list[pd.DataFrame] = []
    for fold in range(5):
        df_cur = main_process(task, site, feature, view=view, order=fold, score_sum=score_sum, filt=filt, name_str=name_str)
        df_list.append(df_cur)
    assert len(df_list)>1 
    renamed_dfs = []
    for i, df in enumerate(df_list, 1):
        df_temp = df.copy()
        df_temp = df_temp.rename(columns={col: f"{col}_fold{i}" for col in df.columns if col.startswith('D')})
        renamed_dfs.append(df_temp)
    df = reduce(lambda left, right: pd.merge(left, right, on=['ID', 'ScanDatum', 'ID_Timepoint'], how='outer'), renamed_dfs)
    column_list = list(df_list[0].columns.values[3:])
    for d in column_list:
        fold_cols = [f"{d}_fold{i}" for i in range(1, 6)]
        df[f"{d}_foldmean"] = df[fold_cols].mean(axis=1)
        df[f"{d}_foldstd"] = df[fold_cols].std(axis=1)

    df = df.sort_values(by='ID').reset_index(drop=True)
    return df


def merge_feature_process(task:Literal['TE', 'CSA', 'ALL'], 
                          site:Literal['Wrist', 'MCP', 'Foot'],
                          view:List[str]=['TRA', 'COR'],
                          score_sum:bool=False, filt:Optional[list]=None,
                          name_str:str='temp'):
    df = None
    for feature in ['TSY','SYN','BME']:
        if not os.path.exists(f'./output/all_te_{name_str}/1foldmerged_{site}_{feature}_{task}_sum{score_sum}.csv'):
            df_cur = merge_fold_process(task, site, feature, view, score_sum, filt, name_str=name_str)
        else:
            df_cur = pd.read_csv(f'./output/all_te_{name_str}/1foldmerged_{site}_{feature}_{task}_sum{score_sum}.csv', index_col=0)
        if score_sum:
            rename_dict = {
                            col: f"{site}_{feature}_{col.replace('_foldmean', '')}_pred"
                            for col in df_cur.columns if col.endswith('_foldmean')
                            }
            df_cur = df_cur.rename(columns=rename_dict)
            # df_cur = df_cur.sort_values(by='ID').reset_index(drop=True)
        if df is None: 
            df = df_cur
        else: 
            df = pd.merge(df, df_cur, on=['ID', 'ScanDatum', 'ID_Timepoint'], how='inner')
    assert df is not None
    df = df.sort_values(by='ID').reset_index(drop=True)

    path = f'./output/all_te_{name_str}/2featuremerged_{site}_{task}_sum{score_sum}.csv'
    df.to_csv(path)
    return df


def merge_site_process(task:Literal['TE', 'CSA', 'ALL'], 
                       view:List[str]=['TRA', 'COR'],
                       score_sum:bool=False, filt:Optional[list]=None,
                       name_str:str='temp'):
    dir = f'./output/all_te_{name_str}'
    if not os.path.exists(dir): 
        os.makedirs(dir)
    df = None
    for site in ['Wrist', 'MCP', 'Foot']:
        if not os.path.exists(f'./output/all_te_{name_str}/2featuremerged_{site}_{task}_sum{score_sum}.csv'):
            df_cur = merge_feature_process(task, site, view, score_sum, filt, name_str)
        else:
            df_cur = pd.read_csv(f'./output/all_te_{name_str}/2featuremerged_{site}_{task}_sum{score_sum}.csv', index_col=0)
        if df is None: df = df_cur
        else: df = pd.merge(df, df_cur, on=['ID', 'ScanDatum', 'ID_Timepoint'], how='outer')
    assert df is not None
    df = df.sort_values(by='ID').reset_index(drop=True)
    path = f'./output/all_te_{name_str}/3sitemerged_{task}_sum{score_sum}.csv'
    df.to_csv(path)
    label_id(path)
    return df


if __name__=='__main__':
    for ss in [False]:  # True, 
        merge_site_process('TE', view=['TRA', 'COR'], score_sum=ss, filt=None, name_str=f'_sum{ss}_with_std_split')



