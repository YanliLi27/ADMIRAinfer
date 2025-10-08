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


import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def main_process(task:Literal['TE', 'CSA', 'EAC', 'ATL', 'ALL'], site:Literal['Wrist', 'MCP', 'Foot'],
                 feature:Literal['TSY','SYN','BME'], 
                 view:List[str]=['TRA', 'COR'],
                 order:int=0, score_sum:bool=False, filt:Optional[list]=None,
                 name_str:str='outside/250825') -> Union[pd.DataFrame, None]:
    if not os.path.exists(f'./output/{name_str}'): 
        os.makedirs(f'./output/{name_str}')
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
        df_cur:Union[pd.DataFrame, None] = main_process(task, site, feature, view=view, order=fold, 
                                                        score_sum=score_sum, filt=filt, name_str=name_str)
        if df is None: df = df_cur
        else: 
            if isinstance(df_cur, pd.DataFrame) and not df_cur.empty: df = pd.concat([df, df_cur])
            else: continue
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

    path = f'./output/{name_str}/1foldmerged_{site}_{feature}_{task}_sum{score_sum}.jpg'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.clf()




    df_res.to_csv(f'./output/{name_str}/1foldmerged_{site}_{feature}_{task}_sum{score_sum}.csv')
    return df_res


def merge_feature_process(task:Literal['TE', 'CSA', 'EAC', 'ATL', 'ALL'], 
                          site:Literal['Wrist', 'MCP', 'Foot'],
                          view:List[str]=['TRA', 'COR'],
                          score_sum:bool=False, filt:Optional[list]=None,
                          name_str:str='outside/250825'):
    df = None
    for feature in ['TSY','SYN','BME']:
        if not os.path.exists(f'./output/{name_str}/1foldmerged_{site}_{feature}_{task}_sum{score_sum}.csv'):
            df_cur = merge_fold_process(task, site, feature, view, score_sum, filt, name_str)
        else:
            df_cur = pd.read_csv(f'./output/{name_str}/1foldmerged_{site}_{feature}_{task}_sum{score_sum}.csv', index_col=0)
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
    if not os.path.exists(f'./output/{name_str}'): 
        os.makedirs(f'./output/{name_str}')
    df = None
    for site in sites:
        if not os.path.exists(f'./output/{name_str}/2featuremerged_{site}_{task}_sum{score_sum}.csv'):
            df_cur = merge_feature_process(task, site, view, score_sum, filt, name_str)
        else:
            df_cur = pd.read_csv(f'./output/{name_str}/2featuremerged_{site}_{task}_sum{score_sum}.csv', index_col=0)
        if df is None: df = df_cur
        else: df = pd.merge(df, df_cur, on=['ID', 'ScanDatum', 'ID_Timepoint'], how='outer')
    assert df is not None
    df = df.sort_values(by='ID').reset_index(drop=True)
    df.to_csv(f'./output/{name_str}/3sitemerged_{task}_sum{score_sum}.csv')
    return df


if __name__=='__main__':
    filt = None
    filt_type = 'Arth'
    assert filt_type in ['Arth', 'Treat', 'Csa', 'Atlas']
    # filt = [2848, 2850, 2854, 2856, 2866, 2870, 2872, 2886, 2891, 2895, 2899, 3006, 3012, 3017, 3021, 3024, 3031,
    #         3061, 3062, 3067, 3078, 3110, 3116, 3118, 3121, 3123, 3124, 3127, 3139, 3145, 3146, 3152, 3153, 3154,
    #         3157, 3159, 3162, 3166, 3170, 3177, 3207, 3217, 3233, 3236, 3240, 3242, 3254, 3261, 3280, 3285, 3306,
    #         3423, 3443, 3448, 3449, 3450, 3459, 3460, 3472]

    # 250902 - for eac follow-ups
    # filt = [2848,2850,2854,2856,2866,2868,2869,2870,2872,2885,2886,2891,2895,2899,3006,3012,3017,3021,3024,3031,3050,
    #         3061,3062,3067,3078,3101,3110,3115,3116,3118,3120,3121,3123,3124,3127,3130,3139,3144,3145,3146,3152,3153,
    #         3154,3157,3159,3162,3166,3167,3170,3177,3178,3183,3184,3188,3194,3196,3197,3206,3207,3209,3210,3212,3214,
    #         3217,3223,3226,3227,3232,3233,3235,3236,3240,3242,3244,3251,3254,3256,3261,3265,3267,3271,3273,3280,3281,
    #         3282,3284,3285,3288,3289,3290,3292,3296,3298,3299,3300,3302,3304,3306,3313,3320,3321,3322,3327,3332,3334,
    #         3338,3340,3342,3344,3347,3348,3349,3360,3361,3363,3364,3376,3377,3379,3380,3381,3386,3387,3388,3389,3391,
    #         3392,3393,3397,3398,3404,3406,3408,3412,3413,3415,3421,3422,3423,3424,3428,3429,3430,3431,3436,3439,3440,
    #         3443,3444,3446,3447,3448,3449,3452,3454,3456,3459,3460,3461,3466,3467,3468,3469,3470,3471,3472,3475,3479,
    #         3482,3484,3493,3494,3498,3500,3501,3507,3512,3517,3521,3522,3523,3527,3528,3529,3533,3536,3538,3543,3544,
    #         3545,3548,3549,3551,3556,3559,3560,3564,3565,3568,3569,3570,3572,3574,3575,3576,3579,3580,3581,3589,3591,
    #         3593,3597,3598,3599,3600,3602,3604,3605,3606,3607,3608,3609,3612,3616,3617,3618,3619,3622,3624,3628,3629,
    #         3630,3631,3632,3633,3634,3638,3639,3640,3648,3649,3655,3657,3658,3660,3661,3662,3665,3674,3675,3678,3679,
    #         3684,3685,3686,3688,3694,3696,3700,3703,3704,3706,3707,3709,3711,3712,3714,3717,3719,3722,3724,3727,3728,
    #         3731,3732,3734,3735,3740,3743,3745,3748,3750,3752,3753,3754,3758,3762,3764,3766,3768,3769,3770,3773,3776,
    #         3778,3779,3786,3790,3793,3794,3795,3796,3801,3803,3806,3811,3812,3814,3817,3820,3821,3828,3829,3830,3836,
    #         3841,3843,3844,3848,3850,3856,3858,3859,3863,3870,3871,3872,3873,3874,3877,3881,3890,3923]
    
    if filt is not None: 
        filt = [filt_type + str(x).zfill(4) for x in filt] if filt_type in ['Arth', 'Treat'] else [filt_type + str(x).zfill(3) for x in filt]
    for ss in [False]:  # True, 
        merge_site_process('ALL', view=['TRA', 'COR'], sites=['Wrist', 'MCP', 'Foot'], score_sum=ss, filt=filt, name_str='outside/251008')



