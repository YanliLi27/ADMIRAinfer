# 从esmira文件夹和ramris里面读取全部的数据，然后用pkl的内容来筛选ID，获得df
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from .headdict import get_score_head
from .central_selector import central_selector
from typing import List, Literal
import pickle


def get_dict() -> dict:
    sites:list=['Wrist', 'MCP', 'Foot']
    views:list=['TRA', 'COR']
    this_dict = dict()
    for site in sites:
        for view in views:
            this_dict[f'{site}_{view}'] = None
    return this_dict


def get_id_from_mri(mri_root:str, groups:list=['EAC','CSA','ATL'], sites:list=['Wrist', 'MCP', 'Foot'], views:list=['TRA', 'COR']) -> pd.DataFrame:
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


def get_id_from_ramris(ramris_root:str, prefix:str) -> pd.DataFrame:
    # CSANUMM 那一列需要左边加Csa, 并且zfill到3
    df:pd.DataFrame = pd.read_csv(ramris_root, sep=';')
    expected_heads = get_score_head(return_all=True)
    for head in expected_heads:
        head1, head2 = head+'.1', head+'.2'
        df[head1] = df[head1].apply(lambda x: process_score(x))
        df[head2] = df[head2].apply(lambda x: process_score(x))
        df[head] = df[[head1, head2]].apply(lambda row: np.nanmean(row) if not all(np.isnan(row)) else np.nan, axis=1)
    df = df.rename(columns={prefix: 'ID'})
    if 'CSA' in prefix: replace:str = 'Csa'
    elif 'EAC' in prefix: replace:str = 'Arth'
    elif 'Atlas' in prefix: replace:str = 'Atlas'
    else: raise AttributeError(f'{prefix}')
    n:int = 4 if 'Arth' in prefix else 3
    df['ID'] = df['ID'].apply(lambda x: replace + str(x).zfill(n))
    target_column = ['ID'] + expected_heads
    return df[target_column].copy()


def obtain_id(name:str):
    # ['CSA_Wrist_TRA\\ESMIRA-LUMC-Csa649_CSA-20180606-RightWrist_PostTRAT1f_0.mha:0to5plus0to7'，
    #         'CSA_Wrist_COR\\ESMIRA-LUMC-Csa649_CSA-20180606-RightWrist_PostCORT1f_0.mha:8to13plus8to15']
    # EAC:  EAC_Wrist_TRA\\ESMIRA-LUMC-Arth4443_EAC-20180103-RightWrist_PostTRAT1f_1.mha:0to5plus0to7
    # CSA:  CSA_Wrist_TRA\\ESMIRA-LUMC-Csa649_CSA-20180606-RightWrist_PostTRAT1f_0.mha:0to5plus0to7
    # ATL:  ATL_Wrist_TRA\\ESMIRA-LUMC-Atlas156_ATL-20140625-RightWrist_PostTRAT1f.mha:0to5plus0to7
    item = name.split('-')[2]
    item = item.split('_')[0]
    return item


def pkl_reader(site:Literal['Wrist','MCP','Foot']='Wrist', 
               feature:Literal['TSY','SYN','BME']='TSY',
               order:int=0,
               sum_score:bool=True):
    # 用pkl_reader把它们转化为csv数据，然后用和此处的dataset相同的形式进行保存
    # 【un22_EAC_CSA_ATL__{site}_2dirc_1.pkl】  
    # 里面存的是 [5split  *[id[path1:cs, path2:cs, path3:cs, ...]]]的list
    # 【un22_EAC_CSA_ATL__{site}_{feature}_2reader_1__sc.pkl 
    # 里面存的是对应BIO的 [5split  *[id[site1_array, site2_array], id[[site1_array, site2_array]]
    path:str = f'sum{sum_score}/un22_EAC_CSA_ATL__{site}_2dirc_1.pkl'
    pkl_dir:str = r'E:\ADMIRA_models\split'
    pkl_path:str = os.path.join(pkl_dir, path)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    data_split:np.ndarray = np.asarray(data[order])
    # [id[path1:cs, path2:cs, path3:cs, ...]] -> np.ndarray
    data_list:list = list(data_split[:, 0])  # [id1-path1:cs, id2-path1:cs, ...]
    # get id
    data_list:list = [obtain_id(item) for item in data_list]
    return data_list


def all_initialization(mri_root:str=r'E:\ESMIRA_RAprediction\Export20Jun22', 
                       ramris_root:List[str]=['CSA', 'EAC', 'ATL'],
                       site:Literal['Wrist','MCP','Foot']='Wrist', 
                       feature:Literal['TSY','SYN','BME']='TSY',
                       order:int=0,
                       sum_score:bool=True) ->pd.DataFrame:
    path_zoo = {'CSA':r'D:\ESMIRA\SPSS data\5. CSA_T1_MRI_scores_SPSS.csv',  # CSANUMM
                  'EAC':r'D:\ESMIRA\SPSS data\1. EAC baseline.csv',  # EACNUMM
                  'ATL':r'D:\ESMIRA\SPSS data\3. Atlas.csv'}  # AtlasNR  1,2...
    prefix_zoo = {'CSA': 'CSANUMM',  # CSANUMM
                  'EAC': 'EACNUMM',  # EACNUMM
                  'ATL': 'AtlasNR'}  # AtlasNR  1,2...
    if not os.path.exists(r'./datasets/all_mri_init.csv'): 
        mri_id_path:pd.DataFrame = get_id_from_mri(mri_root)
        mri_id_path.to_csv(r'./datasets/all_mri_init.csv')
    else:
        mri_id_path = pd.read_csv(r'./datasets/all_mri_init.csv')
        mri_id_path['DATE'] = mri_id_path['DATE'].astype(int)
    # ID (Csa003), DATE(20202020), ID_DATE(ID;DATE), Site_View * 6 (abs_path;NtoN+7)
    if not os.path.exists(r'./datasets/all_ramris_init.csv'): 
        ramris_id_score:pd.DataFrame = pd.DataFrame()
        for root in ramris_root:
            ramris_root_path = path_zoo[root]
            prefix = prefix_zoo[root]
            score:pd.DataFrame = get_id_from_ramris(ramris_root_path, prefix)
            ramris_id_score = pd.concat([ramris_id_score, score])
        ramris_id_score.to_csv(r'./datasets/all_ramris_init.csv')
    else:
        ramris_id_score = pd.read_csv(r'./datasets/all_ramris_init.csv')
    
    # 合并MRI与RAMRIS
    result = pd.merge(mri_id_path, ramris_id_score, on='ID', how='left')
    result['DATE'] = result['DATE'].fillna(0).astype(int)
    result['DATE'] = result['DATE'].replace(0, np.nan)

    # 用pkl的数据进行筛选
    fold_id:list = pkl_reader(site, feature, order, sum_score)
    result:pd.DataFrame = result[result['ID'].isin(fold_id)]
    return result