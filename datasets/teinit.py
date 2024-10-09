# 用于获得te的全部需要计算的列表
# 目标是一个dict，并保存：
# ID, DATE, ID_DATE, Wrist_TRA, Wrist_COR, MCP_TRA, MCP_COR, Foot_TRA, Foot_COR (path;cs), 以及分数
import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
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


def get_id_from_mri(mri_root:str, groups:list=['TRT', 'PLA'], sites:list=['Wrist', 'MCP', 'Foot'], views:list=['TRA', 'COR']) -> pd.DataFrame:
    data = {}
    # TE直接在Drive上面读取的，所以一开始就可以通过路径来获取ID
    id_list = os.listdir(mri_root)  # AIMIRA_Patient_Treatxxxx_TRT
    # R:\AIMIRA\AIMIRA_Database\LUMC\AIMIRA_patient_Treat0001_PLA
    for patient in tqdm(id_list):
        if patient.split('_')[-1] not in groups: continue
        # \20150319
        cur_id = patient.split('_')[-2]  # Treat0001
        dates = os.listdir(os.path.join(mri_root, patient))
        # \LeftWrist_PostTRAT1f\images\itk\AIMIRA-LUMC-Treat0001_PLA-20150319-LeftWrist_PostTRAT1f.mha
        for date in dates:
            cur_id_date = f'{cur_id};{date}'  # Treat0001;20150319
            data[cur_id_date] = get_dict()
            # get the paths in current folder
            all_scan_folder = os.listdir(os.path.join(mri_root, patient, date))
            for site in sites:
                for view in views:
                    matches = [item for item in all_scan_folder if re.search(f'{site}_Post{view}T1f', item)]
                    if matches:
                        data[cur_id_date][f'{site}_{view}'] = central_selector(os.path.join(mri_root, patient, date, matches[0]))
    final_data:pd.DataFrame = pd.DataFrame()
    for id_date in data.keys():
        id, date = id_date.split(';')
        paths:dict = data[id_date]
        paths['ID'], paths['DATE'], paths['ID_DATE'] = id, date, id_date
        final_data.loc(len(final_data)) = paths
        paths.clear()
    return final_data  # ID (Treat0001), DATE (20150319), ID_DATE (Treat0001;20150319), 

def get_id_from_ramris(ramris_root:str) -> pd.DataFrame:
    # CSANUMM 那一列需要左边加Csa, 并且zfill到3
    df:pd.DataFrame = pd.read_csv(ramris_root, sep=';')
    expected_heads = get_score_head(return_all=True)
    for head in expected_heads:
        head1, head2 = head+'.1', head+'.2'
        df[head1] = df[head1].apply(lambda x: np.nan if x > 10 else x)
        df[head2] = df[head2].apply(lambda x: np.nan if x > 10 else x)
        df[head] = df[[head1, head2]].apply(lambda row: np.nanmean(row) if not all(np.isnan(row)) else np.nan, axis=1)
    df.rename(columns={'TENR': 'ID', 'SCANdatum':'DATE', 'hoeveelste_MRI':'TimePoint'})
    df['DATE'] = df['DATE'].apply(lambda x: str(x).replace('-', ''))  # 2015-03-19 -> 20150319
    df['ID'] = df['ID'].apply(lambda x: 'Treat' + str(x).zfill(4))
    target_column = ['ID', 'DATE', 'TimePoint'] + expected_heads
    return df[target_column].copy()


def te_initialization(mri_root:str='R:\\AIMIRA\\AIMIRA_Database\\LUMC', 
                      ramris_root:str='R:\\AIMIRA\\AIMIRA_Scores\\SPSS data\\TE_scores_MRI_serieel_nieuw.csv') -> pd.DataFrame:
    # 首先获得全部的IDlist，根据mri_root进行
    mri_id_path:pd.DataFrame = get_id_from_mri(mri_root)
    # ID (Treat0003), DATE(20202020), ID_DATE(ID;DATE), Site_View * 6 (abs_path;NtoN+7)
    ramris_id_score:pd.DataFrame = get_id_from_ramris(ramris_root)
    # ID (from TENR to ID), Site_Bio_FEATURES * N
    # 直接用pandas自带的合并来合并？原则上这里需要用ID_DATE来合并，但是因为spss没有这个信息所以只能用ID来merge
    result = pd.merge(mri_id_path, ramris_id_score, on='ID', how='outer')
    return result