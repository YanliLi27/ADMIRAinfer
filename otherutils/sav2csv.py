import pandas as pd
import pyreadstat
import re
from collections import defaultdict

def group_anagrams(words):
    groups = defaultdict(list)
    for word in words:
        key = ''.join(sorted(word))
        groups[key].append(word)
    return [group for group in groups.values() if len(group) > 1]


def find_anagrams(s, words):
    key = ''.join(sorted(s))
    return [w for w in words if ''.join(sorted(w)) == key and w != s]

if __name__=="__main__":
    project:dict = \
    {
    'WRLUERO':'WREROLU', 
    'WRULERO':'WREROUL', 
    'WRLUBME':'WRBMELU', 
    'WRULBME':'WRBMEUL', 
    'WRVITSY':'WRTSYVI', 
    'WRIVTSY':'WRTSYIV'
    }

    lib:list = \
    [
    'WRERO1', 'WRERO2', 'WRERO3', 'WRERO4', 'WRERO5', 'WREROTM', 'WREROTD', 'WREROCA', 'WREROHA', 'WREROSC', 
    'WREROLU', 'WREROTQ', 'WREROPI', 'WRERORA', 'WREROUL', 
    'WRBME1', 'WRBME2', 'WRBME3', 'WRBME4', 'WRBME5', 'WRBMETM', 'WRBMETD', 'WRBMECA', 
    'WRBMEHA', 'WRBMESC', 'WRBMELU', 'WRBMETQ', 'WRBMEPI', 'WRBMERA', 'WRBMEUL', 
    'WRSYNRU', 'WRSYNRC', 'WRSYNIC', 
    'WRTSYVI', 'WRTSYV', 'WRTSYIV', 'WRTSYIII', 'WRTSYII', 'WRTSYI', 'WRTSY1', 'WRTSY2', 'WRTSY3', 'WRTSY4',  # wrist
    'MCDERO2', 'MCDERO3', 'MCDERO4', 'MCDERO5', 'MCPERO2', 'MCPERO3', 'MCPERO4', 'MCPERO5', 
    'MCDBME2', 'MCDBME3', 'MCDBME4', 'MCDBME5', 'MCPBME2', 'MCPBME3', 'MCPBME4', 'MCPBME5', 
    'MCSYN2', 'MCSYN3', 'MCSYN4', 'MCSYN5', 
    'MCFTSY2', 'MCFTSY3', 'MCFTSY4', 'MCFTSY5', 'MCETSY2', 'MCETSY3', 'MCETSY4', 'MCETSY5',  # mcp
    'MTDERO1', 'MTDERO2', 'MTDERO3', 'MTDERO4', 'MTDERO5', 'MTPERO1', 'MTPERO2', 'MTPERO3', 'MTPERO4', 'MTPERO5', # MTERO
    'MTDBME1', 'MTDBME2', 'MTDBME3', 'MTDBME4', 'MTDBME5', 'MTPBME1', 'MTPBME2', 'MTPBME3', 'MTPBME4', 'MTPBME5', # MTBME
    'MTSYN1', 'MTSYN2', 'MTSYN3', 'MTSYN4', 'MTSYN5', # MTSYN
    'MTETSY1', 'MTETSY2', 'MTETSY3', 'MTETSY4', 'MTETSY5', 'MTFTSY1', 'MTFTSY2', 'MTFTSY3', 'MTFTSY4', 'MTFTSY5'  # MTTSY 
    ]

    # 读取sav文件
    path_lists = [r'R:\ESMIRA\ESMIRA_Scores\SPSS data\5. CSA_T1_MRI_scores_SPSS.sav',
                r'R:\ESMIRA\ESMIRA_Scores\SPSS data\6. CSA_T2_MRI_scores_SPSS.sav',
                r'R:\ESMIRA\ESMIRA_Scores\SPSS data\7. CSA_T4 MRI_scores_SPSS.sav',  # CSANUMM
                r'R:\ESMIRA\ESMIRA_Scores\SPSS data\1. EAC baseline.sav', 
                r'R:\ESMIRA\ESMIRA_Scores\SPSS data\2. EAC longitudinal.sav',
                r'R:\ESMIRA\ESMIRA_Scores\SPSS data\3. Atlas.sav']
    for idx, file_path in enumerate(path_lists):
        df, meta = pyreadstat.read_sav(file_path)
        # 查看前几行数据  
        print(df.head())
        # 转换为CSV
        csv_path = file_path.replace('.sav', '.csv')


        if idx==4:
            new_columns = []
            for col in df.columns:
                if col in ['WRLUERO', 'WRULERO', 'WRLUBME', 'WRULBME', 'WRVITSY', 'WRIVTSY']:
                    new_columns.append(project[col])
                else:
                    new_cols:list = find_anagrams(col, lib)
                    if len(new_cols)==1:
                        new_columns.append(new_cols[0])
                    elif len(new_cols)>1:
                        raise ValueError(f'{new_cols} are anagrams for {col}')
                    else:
                        new_columns.append(col)  # 如果没有匹配，保持不变
            # 替换列名
            df.columns = new_columns

        df.to_csv(csv_path, index=False, encoding='utf-8')  # utf-8-sig避免中文乱码



        print(f"已成功将 {file_path} 转为 {csv_path}")