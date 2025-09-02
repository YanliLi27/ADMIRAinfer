from typing import Literal, Optional, List, Union
import pandas as pd
import os
import torch
from datasets.getdata_outside import getdata
from models.get_model import getmodel
from trained_weights.get_weight import getweight_outside
from utils.get_head import return_head, return_head_gt
from torch.utils.data import DataLoader
from tqdm import tqdm


def main_process(task:Literal['TE', 'CSA', 'EAC', 'ATL', 'ALL'], site:Literal['Wrist', 'MCP', 'Foot'],
                 feature:Literal['TSY','SYN','BME'], 
                 view:List[str]=['TRA', 'COR'],
                 order:int=0, score_sum:bool=False, filt:Optional[list]=None,
                 name_str:str='outside/250825') -> Union[pd.DataFrame, None]:
    model = getmodel(site, feature, view, score_sum)  # DONE!
    model = getweight_outside(model, r'E:\ADMIRA_models\weights', site, feature, score_sum, view, order)  # DONE!
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    # obtain data
    # 数据读取需要task, site, feature
    # 数据返回应该是 img, scores, path (id, date)
    data, maxidx = getdata(task, site, feature, view, filt, score_sum, order, 'Online', True)
    if data == None: return None
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
    df.to_csv(f'./output/{name_str}/0unmerged_{site}_{feature}_{task}_sum{score_sum}_{order}.csv')
    return df
    # inference:
    # 直接for x,y,z in Dataloader():
    # pred = model(x)
    # 根据z划分出ID和time_point(只需要有时间就可以了，
    # 再存一个ID_TP: CSA001_20140901用来合并
    # 存到一个csv里面

def merge_fold_process(task:Literal['TE', 'CSA', 'EAC', 'ATL', 'ALL'], site:Literal['Wrist', 'MCP', 'Foot'],
                 feature:Literal['TSY','SYN','BME'], 
                 view:List[str]=['TRA', 'COR'],
                 score_sum:bool=False, filt:Optional[list]=None,
                 name_str:str='outside/250825'):
    df = None
    for fold in range(5):
        df_cur:Union[pd.DataFrame, None] = main_process(task, site, feature, view=view, order=fold, score_sum=score_sum, filt=filt)
        if df is None: df = df_cur
        else: 
            if isinstance(df_cur, pd.DataFrame) and not df_cur.empty: df = pd.concat([df, df_cur])
            else: continue
    assert df is not None
    df = df.sort_values(by='ID').reset_index(drop=True)
    column_list = list(df.columns.values[3:])
    df_res = df.groupby(['ID', 'ScanDatum', 'ID_Timepoint'], as_index=False)[column_list].mean()
    df_res.to_csv(f'./output/{name_str}/1foldmerged_{site}_{feature}_{task}_sum{score_sum}.csv')
    return df_res


def merge_feature_process(task:Literal['TE', 'CSA', 'EAC', 'ATL', 'ALL'], 
                          site:Literal['Wrist', 'MCP', 'Foot'],
                          view:List[str]=['TRA', 'COR'],
                          score_sum:bool=False, filt:Optional[list]=None,
                          name_str:str='outside/250825'):
    df = None
    for feature in ['TSY','SYN','BME']:
        if not os.path.exists(f'./output/all_te/1foldmerged_{site}_{feature}_{task}_sum{score_sum}.csv'):
            df_cur = merge_fold_process(task, site, feature, view, score_sum, filt)
        else:
            df_cur = pd.read_csv(f'./output/all_te/1foldmerged_{site}_{feature}_{task}_sum{score_sum}.csv', index_col=0)
        if score_sum:
            df_cur = df_cur.rename(columns={'sums': f'{site}_{feature}_pred', 'sums_gt': f'{site}_{feature}_gt'})
            # df_cur = df_cur.sort_values(by='ID').reset_index(drop=True)
        if df is None: df = df_cur
        else: df = pd.merge(df, df_cur, on=['ID', 'ScanDatum', 'ID_Timepoint'], how='inner')
    assert df is not None
    df = df.sort_values(by='ID').reset_index(drop=True)
    df.to_csv(f'./output//{name_str}/2featuremerged_{site}_{task}_sum{score_sum}.csv')
    return df


def merge_site_process(task:Literal['TE', 'CSA', 'EAC', 'ATL', 'ALL'], 
                       view:List[str]=['TRA', 'COR'],
                       sites:List[str]=['Wrist', 'MCP', 'Foot'],
                       score_sum:bool=False, filt:Optional[list]=None,
                       name_str:str='outside/250825'):
    df = None
    for site in sites:
        if not os.path.exists(f'./output/all_te/2featuremerged_{site}_{task}_sum{score_sum}.csv'):
            df_cur = merge_feature_process(task, site, view, score_sum, filt)
        else:
            df_cur = pd.read_csv(f'./output/all_te/2featuremerged_{site}_{task}_sum{score_sum}.csv', index_col=0)
        if df is None: df = df_cur
        else: df = pd.merge(df, df_cur, on=['ID', 'ScanDatum', 'ID_Timepoint'], how='outer')
    assert df is not None
    df = df.sort_values(by='ID').reset_index(drop=True)
    df.to_csv(f'./output//{name_str}/3sitemerged_{task}_sum{score_sum}.csv')
    return df


if __name__=='__main__':
    # filt = None
    filt_type = 'Arth'
    assert filt_type in ['Arth', 'Treat', 'Csa', 'Atlas']
    filt = [2848, 2850, 2854, 2856, 2866, 2870, 2872, 2886, 2891, 2895, 2899, 3006, 3012, 3017, 3021, 3024, 3031,
            3061, 3062, 3067, 3078, 3110, 3116, 3118, 3121, 3123, 3124, 3127, 3139, 3145, 3146, 3152, 3153, 3154,
            3157, 3159, 3162, 3166, 3170, 3177, 3207, 3217, 3233, 3236, 3240, 3242, 3254, 3261, 3280, 3285, 3306,
            3423, 3443, 3448, 3449, 3450, 3459, 3460, 3472]
    filt = [filt_type + str(x).zfill(4) for x in filt] if filt_type in ['Arth', 'Treat'] else [filt_type + str(x).zfill(3) for x in filt]
    for ss in [False]:  # True, 
        merge_site_process('ALL', view=['TRA', 'COR'], sites=['Wrist'], score_sum=ss, filt=filt, name_str='outside/250825')



