from typing import Literal, Optional, List
import pandas as pd
import os
import torch
from datasets.get_data_preconstrast import getdata_pre
from models.get_model import getmodel
from trained_weights.get_weight import getweight_outside
from utils.get_head import return_head, return_head_gt
from torch.utils.data import DataLoader
from tqdm import tqdm

from otherutils.label_id_function import label_id

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def main_process(task:Literal['TE'], site:Literal['Wrist', 'MCP', 'Foot'],
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
    data, maxidx = getdata_pre(task, site, feature, view, filt, score_sum, True)
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
        res_head.extend(['sums', 'sums_gt'])
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
    df = None
    for fold in range(5):
        df_cur = main_process(task, site, feature, view=view, order=fold, score_sum=score_sum, filt=filt, name_str=name_str)
        if df is None: df = df_cur
        else: df = pd.concat([df, df_cur])
    assert df is not None
    df = df.sort_values(by='ID').reset_index(drop=True)
    column_list = list(df.columns.values[3:])
    key_cols = ['ID', 'ScanDatum', 'ID_Timepoint']
    pred_cols:list = [col for col in column_list if 'GT' not in col and 'gt' not in col]
    gt_cols:list = [col for col in column_list if 'GT' in col or 'gt' in col]

    # get sum 
    df[f'{site}_{feature}_predscore_sum'] = df[pred_cols].sum(axis=1)
    df[f'{site}_{feature}_gt_sum'] = df[gt_cols].sum(axis=1)
    df[f'{site}_{feature}_diff_for_analysis'] = df[f'{site}_{feature}_predscore_sum'] - df[f'{site}_{feature}_gt_sum'] 

    # df ['ID', 'ScanDatum', 'ID_Timepoint', 'f'{site}_{feature}_predscore_sum'', 'gt_sum_for_analysis', 'diff']

    extended_column_list = column_list + [f'{site}_{feature}_predscore_sum', 
                                          f'{site}_{feature}_gt_sum', 
                                          f'{site}_{feature}_diff_for_analysis']
    df_res = df.groupby(key_cols, as_index=False)[extended_column_list].mean()

    df_res[f'{site}_{feature}_diff'] = df_res[f'{site}_{feature}_predscore_sum'] - df_res[f'{site}_{feature}_gt_sum']
    df_res[f'{site}_{feature}_abs_diff'] = df_res[f'{site}_{feature}_diff'].apply(lambda x: abs(x))
    # df_res['diff'] = df_res['diff_for_analysis']
    # df_res['abs_diff'] = df_res['diff'].apply(lambda x: abs(x))
    
    df_std:pd.DataFrame = df.groupby(key_cols, as_index=False)[f'{site}_{feature}_predscore_sum'].std()
    df_std:pd.DataFrame = df_std.rename(columns={f'{site}_{feature}_predscore_sum': f'{site}_{feature}_predscore_fold_std'})

    df_res = pd.merge(df_res, df_std, on=['ID', 'ScanDatum', 'ID_Timepoint'], how='left')

    # create plot
    x, y, abs_x = df_res[f'{site}_{feature}_diff'].to_numpy(), \
                  df_res[f'{site}_{feature}_predscore_fold_std'].to_numpy(), \
                  df_res[f'{site}_{feature}_abs_diff'].to_numpy()
    min_val, max_val = min(min(x), min(y)), max(max(x), max(y), -min(x), -min(y))
 
    corr, p_value = pearsonr(y, abs_x)# spearmanr(y, abs_x) # 
    plt.clf()
    plt.scatter(y, x, color='blue', marker='o')
    plt.xlim(0, max_val*2)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.ylabel('diff between pred and gt')
    plt.xlabel('std among models')
    plt.title(f'{site}_{feature}_{task}_sum{score_sum}: corr with abs - {corr}, p - {p_value}')
    plt.grid(True)

    path = f'./output/all_te_{name_str}/1foldmerged_{site}_{feature}_{task}_sum{score_sum}.jpg'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.clf()
    
    df_res.to_csv(path.replace('.jpg', '.csv'))
    return df_res


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
            df_cur = df_cur.rename(columns={'sums': f'{site}_{feature}_pred', 'sums_gt': f'{site}_{feature}_gt'})
            # df_cur = df_cur.sort_values(by='ID').reset_index(drop=True)
        if df is None: df = df_cur
        else: df = pd.merge(df, df_cur, on=['ID', 'ScanDatum', 'ID_Timepoint'], how='inner')
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
    filt = [15, 72, 79, 348, 400, 471]
    for ss in [False]:  # True, 
        merge_site_process('TE', view=['TRA', 'COR'], score_sum=ss, filt=filt, name_str=f'_sum{ss}_with_std_pre')



