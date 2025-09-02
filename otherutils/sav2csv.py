import pandas as pd
import pyreadstat
import re
from collections import defaultdict
from projection_table import get_projection


def group_anagrams(words):
    groups = defaultdict(list)
    for word in words:
        key = ''.join(sorted(word))
        groups[key].append(word)
    return [group for group in groups.values() if len(group) > 1]


def find_anagrams(s, words):
    key = ''.join(sorted(s))
    return [w for w in words if ''.join(sorted(w)) == key and w != s]


def head_regularizer(df:pd.DataFrame):
    # CAS      EAC      ATL     TE
    # CSANUMM, EACNUMM, AtalasNR, TENR -> ID
    # MRI_datum.1, datum_MRI/Datum, DatumMRI,  SCANdatum -> DATE
    # Visitenr, mritijdspunt_num, XXX, hoeveelste_MRI -> TimePoint
    if ('CSANUMM' in df.columns) and ('EACNUMM' not in df.columns):
        df = df.rename(columns={'CSANUMM': 'ID', 'MRI_datum.1':'DATE', 'Visitenr':'TimePoint'})
        df['ID'] = df['ID'].apply(lambda x: 'Csa' + str(int(x)).zfill(3))

        df['DATE'] = df['DATE'].apply(lambda x: str(x).replace('-', ''))  # 2015-03-19 -> 20150319
        if 'TimePoint' not in df.columns:
            df['TimePoint'] = 1

    elif 'EACNUMM' in df.columns:
        if 'datum_MRI' in df.columns:
            df = df.rename(columns={'EACNUMM': 'ID', 'datum_MRI':'DATE'})
            df['TimePoint'] = 1
        elif 'Datum' in df.columns:
            df = df.rename(columns={'EACNUMM': 'ID', 'Datum':'DATE', 'mritijdspunt_num':'TimePoint'})
        df['ID'] = df['ID'].apply(lambda x: 'Arth' + str(int(x)).zfill(4))

        df['DATE'] = df['DATE'].apply(lambda x: str(x).replace('-', ''))  # 2015-03-19 -> 20150319
        if 'TimePoint' not in df.columns:
            df['TimePoint'] = 1

    elif 'AtlasNR' in df.columns:
        df = df.rename(columns={'AtlasNR': 'ID', 'DatumMRI':'DATE'})
        df['TimePoint'] = 1
        df['ID'] = df['ID'].apply(lambda x: 'Atlas' + str(int(x)).zfill(3))

        df['DATE'] = df['DATE'].apply(lambda x: str(x).replace('-', ''))  # 2015-03-19 -> 20150319

    elif 'TENR' in df.columns:
        df = df.rename(columns={'TENR': 'ID', 'SCANdatum':'DATE', 'hoeveelste_MRI':'TimePoint'})
        df['DATE'] = df['DATE'].apply(lambda x: str(x).replace('-', ''))  # 2015-03-19 -> 20150319
        df['ID'] = df['ID'].apply(lambda x: 'Treat' + str(x).zfill(4))
    
    df['ID_DATE'] = df['ID'] + '&' + df['DATE']
    df['TimePoint'] = df['TimePoint'].astype(int)

    for item in ['CSANUMM', 'EACNUMM', 'AtlasNR', 'TENR']:
        if item in df.columns:
            df = df.drop(item, axis=1)
    return df   # ID, DATE, ID_DATE(ID;DATE), TimePoint



if __name__=="__main__":
    projection:dict = get_projection(False)
    projection_cnt:dict = {}
    for key in projection.keys():
        projection_cnt[key] = 1
    # 读取sav文件
    path_lists = [r'R:\ESMIRA\ESMIRA_Scores\SPSS data\CSA BASELINE MRI FILE 27062022.sav',
                r'R:\ESMIRA\ESMIRA_Scores\SPSS data\CSA BASELINE AND FOLLOW-UP MRI FILE May2022_repeatedMRI_long_arthritis_censoring_date.sav',
                r'R:\ESMIRA\ESMIRA_Scores\SPSS data\EAC BASELINE MRI FILE April 2022.sav',  # CSANUMM
                r'R:\ESMIRA\ESMIRA_Scores\SPSS data\EAC FOLLOW-UP MRI FILE August 2025.sav', 
                r'R:\ESMIRA\ESMIRA_Scores\SPSS data\3. Atlas.sav']
    
    # check the date format:
    # for idx, file_path in enumerate(path_lists):
    #     df, meta = pyreadstat.read_sav(file_path)
    #     # 查看前几行数据  
    #     print(df.head())

    for idx, file_path in enumerate(path_lists):
        df, meta = pyreadstat.read_sav(file_path)
        # 查看前几行数据  
        print(df.head(1))
        # 转换为CSV
        csv_path = file_path.replace('.sav', '.csv')


        if 'EAC FOLLOW-UP' in file_path:
            idx = df.columns.get_loc('ERO_informulier_MTP')
            df = df.iloc[:, :idx]
            print('Replace names')
            new_columns = []
            for col in df.columns:
                if col in projection.keys():
                    new_columns.append(projection[col])
                    projection_cnt[col] -= 1
                else:
                    new_columns.append(col)  # 如果没有匹配，保持不变
            assert sum(projection_cnt.values())==0
            # 替换列名
            df.columns = new_columns
        
        print(df.head(1))

        # head regularizer
        print(file_path)
        df = head_regularizer(df)
        print(df.head(1))

        df.to_csv(csv_path, index=False, encoding='utf-8')  # utf-8-sig避免中文乱码



        print(f"已成功将 {file_path} 转为 {csv_path}")
        