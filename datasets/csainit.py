# 获取全部需要计算的列表
# 目标是一个dict，并保存：
# ID, DATE, ID_DATE, Wrist_TRA, Wrist_COR, MCP_TRA, MCP_COR, Foot_TRA, Foot_COR (path;cs)
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from .headdict import get_score_head
from .central_selector import central_selector


def get_dict() -> dict:
    sites:list=['Wrist', 'MCP', 'Foot']
    views:list=['TRA', 'COR']
    this_dict = dict()
    for site in sites:
        for view in views:
            this_dict[f'{site}_{view}'] = None
    return this_dict


def get_id_from_mri(mri_root:str, groups:list=['CSA'], sites:list=['Wrist', 'MCP', 'Foot'], views:list=['TRA', 'COR']) -> pd.DataFrame:
    data = {}
    # 一步到位，直接用一个dict来解决问题，最后把dict展平为df
    for site in sites:
        for group in groups:
            for view in views:
                foldername = os.path.join(mri_root, f'{group}_{site}_{view}')  # E:\ESMIRA_RAprediction\Export20Jun22\EAC_Wrist_TRA
                dir_list = os.listdir(foldername)  # [ESMIRA-LUMC-Arth4161_EAC-20160329-RightWrist_PostTRAT1f_1.mha, ...]
                for item in tqdm(dir_list):
                    abs_path = os.path.join(foldername, item)
                    scatter = item.split('-')
                    cur_id, cur_date = scatter[2].split('_')[0], scatter[3]  # Arth4161, 20160329
                    cur_id_date = f'{cur_id};{cur_date}'  #  Arth4161;20160329 | Csa003;20120411
                    if cur_id_date not in data:  # create the dict
                        data[cur_id_date] = get_dict()
                    data[cur_id_date][f'{site}_{view}'] = central_selector(abs_path) # 'abs_path;2to9'
    # 此时所有的id全部都建立了dict，以'{cur_id};{cur_date}'为key，内部是字典{f'{site}_{view}'}
    # 需要展平
    heads = {'ID':[], 'DATE':[], 'ID_DATE':[]}
    for site in sites:
        for view in views:
            heads[f'{site}_{view}'] = []
    final_data:pd.DataFrame = pd.DataFrame(index=range(len(data.keys())), columns=heads)
    idx = 0
    for id_date in data.keys():
        id, date = id_date.split(';')
        paths:dict = data[id_date]
        paths['ID'], paths['DATE'], paths['ID_DATE'] = id, date, id_date
        final_data.loc[idx] = paths
        idx+=1
        paths.clear()
    return final_data


def process_score(x):
    if isinstance(x, str):
        return int(x) if x.isdigit() and int(x) <= 10 and int(x)>=0 else np.nan
    elif isinstance(x, int):
        return int(x) if int(x) <= 10 and int(x)>=0 else np.nan
    raise ValueError(f'x: {x}')


def get_id_from_ramris(ramris_root:str) -> pd.DataFrame:
    # CSANUMM 那一列需要左边加Csa, 并且zfill到3
    df:pd.DataFrame = pd.read_csv(ramris_root, sep=';')
    expected_heads = get_score_head(return_all=True)
    for head in expected_heads:
        head1, head2 = head+'.1', head+'.2'
        df[head1] = df[head1].apply(lambda x: process_score(x))
        df[head2] = df[head2].apply(lambda x: process_score(x))
        df[head] = df[[head1, head2]].apply(lambda row: np.nanmean(row) if not all(np.isnan(row)) else np.nan, axis=1)
    df = df.rename(columns={'CSANUMM': 'ID'})
    df['ID'] = df['ID'].apply(lambda x: 'Csa' + str(x).zfill(3))
    target_column = ['ID'] + expected_heads
    return df[target_column].copy()


def csa_initialization(mri_root:str=r'E:\ESMIRA_RAprediction\Export20Jun22', 
                       ramris_root:str=r'D:\ESMIRA\SPSS data\5. CSA_T1_MRI_scores_SPSS.csv') -> pd.DataFrame:
    # 首先获得全部的IDlist，根据mri_root进行
    if not os.path.exists(r'./datasets/csa_mri_init.csv'): 
        mri_id_path:pd.DataFrame = get_id_from_mri(mri_root)
        mri_id_path.to_csv(r'./datasets/csa_mri_init.csv')
    else:
        mri_id_path = pd.read_csv(r'./datasets/csa_mri_init.csv')
    # ID (Csa003), DATE(20202020), ID_DATE(ID;DATE), Site_View * 6 (abs_path;NtoN+7)
    if not os.path.exists(r'./datasets/csa_ramris_init.csv'): 
        ramris_id_score:pd.DataFrame = get_id_from_ramris(ramris_root)
        ramris_id_score.to_csv(r'./datasets/csa_ramris_init.csv')
    else:
        ramris_id_score = pd.read_csv(r'./datasets/csa_mri_init.csv')
    # ID (from CSANUMM to ID), Site_Bio_FEATURES * N
    result = pd.merge(mri_id_path, ramris_id_score, on='ID', how='outer')
    return result