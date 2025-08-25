import os
import numpy as np
import csv
import pandas as pd
from typing import Literal, Optional, List, Union
from tqdm import tqdm
import pickle
import re

# 从esmira文件夹和ramris里面读取全部的数据，然后用pkl的内容来筛选ID，获得df
from .dataset import CLIPDataset3D
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


def get_id_from_mri(groups:list=['EAC','CSA','ATL'], sites:list=['Wrist   ', 'MCP', 'Foot'], views:list=['TRA', 'COR'],
                    mode:Literal['Offline', 'Online']='Offline') -> pd.DataFrame:
    if mode == 'Offline': 
        mri_root = r'E:\ESMIRA_RAprediction\Export20Jun22'
        print(f'Offline loading MRIs from {mri_root}, print from getdata_outside.py')
        return get_id_from_mri_offline(mri_root, groups, sites, views)
    elif mode == 'Online': 
        mri_root = r'R:\ESMIRA\ESMIRA_Database\LUMC'
        print(f'Online loading MRIs from {mri_root}, print from getdata_outside.py')
        return get_id_from_mri_online(mri_root, groups, sites, views)
    else: raise AttributeError(f'mode {mode} not supported')


def get_id_from_mri_offline(mri_root:str, groups:list=['EAC','CSA','ATL'], sites:list=['Wrist', 'MCP', 'Foot'], views:list=['TRA', 'COR']) -> pd.DataFrame:
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


def get_id_from_mri_online(mri_root:str, groups:list=['EAC','CSA','ATL'], sites:list=['Wrist', 'MCP', 'Foot'], views:list=['TRA', 'COR']) -> pd.DataFrame:
    # r'R:\ESMIRA\ESMIRA_Database\LUMC' or r'R:\\AIMIRA\\AIMIRA_Database\\LUMC',
    data = {}
    # Online (ALL,TE)直接在Drive上面读取的，所以一开始就可以通过路径来获取ID
    id_list = os.listdir(mri_root)  # ESMIRA_patient_Csa843_CSA(_Arth4431_EAC), AIMIRA_Patient_Treatxxxx_TRT
    # R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Csa843_CSA(_Arth4431_EAC),  R:\AIMIRA\AIMIRA_Database\LUMC\AIMIRA_patient_Treat0001_PLA
    for patient in tqdm(id_list):
        if patient.split('_')[-1] not in groups: continue
        # \20150319
        cur_id = patient.split('_')[-2]  # Treat0001
        dates = os.listdir(os.path.join(mri_root, patient))
        # \LeftWrist_PostTRAT1f\images\itk\AIMIRA-LUMC-Treat0001_PLA-20150319-LeftWrist_PostTRAT1f.mha
        for date in dates:
            if not date.isdigit(): continue
            cur_id_date = f'{cur_id};{date}'  # Treat0001;20150319
            data[cur_id_date] = get_dict()
            # get the paths in current folder
            all_scan_folder = os.listdir(os.path.join(mri_root, patient, date))
            for site in sites:
                for view in views:
                    matches = [item for item in all_scan_folder if re.search(f'{site}_Post{view}T1f', item)]
                    if matches:
                        abs_path = os.path.join(mri_root, patient, date, matches[0], 'images', 'itk')
                        filenames = os.listdir(abs_path)
                        for filename in filenames:
                            if filename[-4:]=='.mha':
                                data[cur_id_date][f'{site}_{view}'] = central_selector(os.path.join(abs_path, filename))
                                break

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
    return final_data  # ID (Treat0001), DATE (20150319), ID_DATE (Treat0001;20150319), 


def process_score(x):
    if isinstance(x, str):
        return int(x) if x.isdigit() and int(x) <= 10 and int(x)>=0 else np.nan
    elif isinstance(x, int):
        return int(x) if int(x) <= 10 and int(x)>=0 else np.nan
    elif isinstance(x, float):
        return int(x) if (not np.isnan(x) and int(x) <= 10 and int(x)>=0) else np.nan
    raise ValueError(f'x: {x}, type: {type(x)}')


def auto_delimiter(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = f.read(2048)  # 读一点样本
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';'])
            delimiter = dialect.delimiter
        except csv.Error:
            # Sniffer失败时，手动兜底
            comma_count = sample.count(',')
            semicolon_count = sample.count(';')
            delimiter = ',' if comma_count >= semicolon_count else ';'
    print(f"Detect the sep: '{delimiter}'")
    return delimiter


def read_single_csv(ramris_root:str, prefix:str):
    delimiter = auto_delimiter(ramris_root)
    df:pd.DataFrame = pd.read_csv(ramris_root, sep=delimiter)
    expected_heads = get_score_head(return_all=True)    

    processed = {}
    for head in expected_heads:
        head1, head2 = head+'.1', head+'.2'
        processed[head1] = df[head1].apply(process_score)
        processed[head2] = df[head2].apply(process_score)
        processed[head] = np.nanmean(
            [processed[head1], processed[head2]], axis=0
        )
    for col, series in processed.items():
        df[col] = series

    df = df.rename(columns={prefix: 'ID'})
    if 'CSA' in prefix: replace:str = 'Csa'
    elif 'EAC' in prefix: replace:str = 'Arth'
    elif 'Atlas' in prefix: replace:str = 'Atlas'
    else: raise AttributeError(f'{prefix}')
    n:int = 4 if 'Arth' in prefix else 3
    df['ID'] = df['ID'].apply(lambda x: replace + str(x).zfill(n))
    target_column = ['ID'] + expected_heads
    return df[target_column].copy()


def get_id_from_ramris(ramris_root:Union[str, list], prefix:str) -> pd.DataFrame:
    # RAMRIS正常读取，只是可能需要修改读的位置
    if isinstance(ramris_root, str):
        return read_single_csv(ramris_root, prefix)
    elif isinstance(ramris_root, list):
        df_com = []
        for root in ramris_root:
            df_com.append(read_single_csv(root, prefix))
        return pd.concat(df_com, ignore_index=True).copy()
    

def obtain_id(name:str):
    # ['CSA_Wrist_TRA\\ESMIRA-LUMC-Csa649_CSA-20180606-RightWrist_PostTRAT1f_0.mha:0to5plus0to7'，
    #         'CSA_Wrist_COR\\ESMIRA-LUMC-Csa649_CSA-20180606-RightWrist_PostCORT1f_0.mha:8to13plus8to15']
    # EAC:  EAC_Wrist_TRA\\ESMIRA-LUMC-Arth4443_EAC-20180103-RightWrist_PostTRAT1f_1.mha:0to5plus0to7
    # CSA:  CSA_Wrist_TRA\\ESMIRA-LUMC-Csa649_CSA-20180606-RightWrist_PostTRAT1f_0.mha:0to5plus0to7
    # ATL:  ATL_Wrist_TRA\\ESMIRA-LUMC-Atlas156_ATL-20140625-RightWrist_PostTRAT1f.mha:0to5plus0to7
    item = name.split('-')[2]  # Arth4443_EAC \Csa649_CSA \Atlas156_ATL
    item = item.split('_')[0]
    return item  # Arth4443  Csa649   Atlas156


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


def reverse_pkl_reader(site:Literal['Wrist','MCP','Foot']='Wrist', 
               feature:Literal['TSY','SYN','BME']='TSY',
               order:int=0,
               sum_score:bool=True,
               full_id:np.ndarray=None):
    # 用pkl_reader把它们转化为csv数据，然后用和此处的dataset相同的形式进行保存
    # 【un22_EAC_CSA_ATL__{site}_2dirc_1.pkl】  
    # 里面存的是 [5split  *[id[path1:cs, path2:cs, path3:cs, ...]]]的list
    # 【un22_EAC_CSA_ATL__{site}_{feature}_2reader_1__sc.pkl 
    # 里面存的是对应BIO的 [5split  *[id[site1_array, site2_array], id[[site1_array, site2_array]]
    assert full_id is not None
    path:str = f'sum{sum_score}/un22_EAC_CSA_ATL__{site}_2dirc_1.pkl'
    pkl_dir:str = r'E:\ADMIRA_models\split'
    pkl_path:str = os.path.join(pkl_dir, path)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    data_revsere:list = []
    for idx in range(5):
        if idx!=order: data_revsere.extend(data[idx])
    data_split:np.ndarray = np.asarray(data_revsere)
    # [id[path1:cs, path2:cs, path3:cs, ...]] -> np.ndarray
    data_list:list = list(data_split[:, 0])  # [id1-path1:cs, id2-path1:cs, ...]
    # get id
    data_list:list = [str(obtain_id(item)) for item in data_list]
    
    full_id = np.asarray(full_id, dtype=str)
    # reverse filt
    filtered_list = list(full_id[~np.isin(full_id, data_list)])

    return filtered_list


def data_initialization(ramris_root:List[str]=['CSA', 'EAC', 'ATL'],
                       loading_mode:Literal['Offline', 'Online']='Offline') ->pd.DataFrame:
    if loading_mode=='Offline':
        path_zoo = {'CSA':r'D:\ESMIRA\SPSS data\5. CSA_T1_MRI_scores_SPSS.csv',  # CSANUMM
                    'EAC':r'D:\ESMIRA\SPSS data\1. EAC baseline.csv',  # EACNUMM
                    'ATL':r'D:\ESMIRA\SPSS data\3. Atlas.csv'}  # AtlasNR  1,2...
    elif loading_mode=='Online':
        path_zoo = {'CSA':[r'R:\ESMIRA\ESMIRA_Scores\SPSS data\5. CSA_T1_MRI_scores_SPSS.csv',  # CSANUMM
                           r'R:\ESMIRA\ESMIRA_Scores\SPSS data\6. CSA_T2_MRI_scores_SPSS.csv',
                           r'R:\ESMIRA\ESMIRA_Scores\SPSS data\7. CSA_T4 MRI_scores_SPSS.csv'],  # why no _ between T4 MRI
                    'EAC':[r'R:\ESMIRA\ESMIRA_Scores\SPSS data\1. EAC baseline.csv', 
                           r'R:\ESMIRA\ESMIRA_Scores\SPSS data\2. EAC longitudinal.csv'],  # EACNUMM
                    'ATL':r'R:\ESMIRA\ESMIRA_Scores\SPSS data\3. Atlas.csv'}  # AtlasNR  1,2...
    prefix_zoo = {'CSA': 'CSANUMM',  # CSANUMM
                  'EAC': 'EACNUMM',  # EACNUMM
                  'ATL': 'AtlasNR'}  # AtlasNR  1,2...
    if not os.path.exists(f'./datasets/intermediate/csv/all_mri_init_{loading_mode}.csv'): 
        mri_id_path:pd.DataFrame = get_id_from_mri(mode=loading_mode)
        mri_id_path.to_csv(f'./datasets/intermediate/csv/all_mri_init_{loading_mode}.csv')
    else:
        mri_id_path = pd.read_csv(f'./datasets/intermediate/csv/all_mri_init_{loading_mode}.csv')
        mri_id_path['DATE'] = mri_id_path['DATE'].astype(int)
    # ID (Csa003), DATE(20202020), ID_DATE(ID;DATE), Site_View * 6 (abs_path;NtoN+7)
    if not os.path.exists(f'./datasets/intermediate/csv/all_ramris_init_{loading_mode}.csv'): 
        ramris_id_score:pd.DataFrame = pd.DataFrame()
        for root in ramris_root:
            ramris_root_path = path_zoo[root]
            prefix = prefix_zoo[root]
            score:pd.DataFrame = get_id_from_ramris(ramris_root_path, prefix)
            ramris_id_score = pd.concat([ramris_id_score, score])
        ramris_id_score.to_csv(f'./datasets/intermediate/csv/all_ramris_init_{loading_mode}.csv')
    else:
        ramris_id_score = pd.read_csv(f'./datasets/intermediate/csv/all_ramris_init_{loading_mode}.csv')
    
    # 合并MRI与RAMRIS
    result = pd.merge(mri_id_path, ramris_id_score, on='ID', how='left')
    result['DATE'] = result['DATE'].fillna(0).astype(int)
    result['DATE'] = result['DATE'].replace(0, np.nan)

    # result.to_csv(r'./datasets/all_init.csv')
    return result


def getdata(task:Literal['CSA', 'TE', 'ATL', 'EAC', 'ALL'], site:Literal['Wrist','MCP','Foot'], feature:Literal['TSY','SYN','BME'], 
            view:list[str]=['TRA', 'COR'], filt:Optional[list]=None, score_sum:bool=False, order:int=0, 
            loading_mode:Literal['Offline', 'Online']='Offline', path_flag:bool=True):
    path_default = {'ALL':f'./datasets/intermediate/csv/all_init_{loading_mode}.csv', 
                    'EAC':f'./datasets/intermediate/csv/eac_init_{loading_mode}.csv',
                    'CSA':f'./datasets/intermediate/csv/csa_init_{loading_mode}.csv', 
                    'ATL':f'./datasets/intermediate/csv/atl_init_{loading_mode}.csv',
                    'TE':f'./datasets/intermediate/csv/te_init_{loading_mode}.csv'}
    paths = path_default[task] if task in ['CSA', 'ATL', 'EAC', 'ALL', 'TE'] else f'./datasets/intermediate/csv/all_init_{loading_mode}.csv'

    # ---------------------------- get the selected rows ----------------------------
    if not os.path.exists(paths):
        df:pd.DataFrame = data_initialization(loading_mode=loading_mode)
        df.to_csv(paths)
    else:
        df:pd.DataFrame = pd.read_csv(paths)

    # ---------------------------- get the selected rows ----------------------------
    # select certain order group using pickle record
    if order==-1:
        pass
    elif order==0:
        fold_id:list = reverse_pkl_reader(site, feature, order, score_sum, df['ID'].values)
        df:pd.DataFrame = df[df['ID'].isin(fold_id)]
    else:
        fold_id:list = pkl_reader(site, feature, order, score_sum)
        df:pd.DataFrame = df[df['ID'].isin(fold_id)]
    
    # exclude by group
    if task in ['CSA', 'EAC', 'ATL', 'TE']:
        short = {'CSA':'Csa', 'EAC':'Arth', 'ATL':'Atlas', 'TE':'Treat'}
        df = df[df['ID'].str.contains(short[task], na=False)]

    # exclude through filt []
    if filt: df = df[df['ID'].isin(filt)]
    df = df.dropna()
    # filt = ['Csa' + str(x).zfill(3) for x in csa_filt] if task=='CSA' else ['Arth' + str(x).zfill(4) for x in eac_filt]
    # CsaXXX or ArthXXXX or AtlasXXX or TreatXXXX
    
    # ---------------------------- get the selected columns ----------------------------
    if len(view)>1:
        path_column = [col for col in df.columns if f'{site}_' in col]
    else:
        path_column = [col for col in df.columns if f'{site}_{view[0]}' in col]
    pre = ['ID', 'DATE', 'ID_DATE']
    pre.extend(path_column)
    score_column = get_score_head(site, feature)
    pre.extend(score_column)
    target:pd.DataFrame = df[pre]    # ID, DA.dropna()TE, ID_DATE, SITE_TRA, SITE_COR   

    # ------------------------------------ return dataset ------------------------------------
    return CLIPDataset3D(target, path_column, score_column, score_sum, path_flag), target.shape[0]
