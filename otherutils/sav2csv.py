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

if __name__=="__main__":
    projection:dict = get_projection(False)
    projection_cnt:dict = {}
    for key in projection.keys():
        projection_cnt[key] = 1
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
                if col in projection.keys():
                    new_columns.append(projection[col])
                    projection_cnt[col] -= 1
                else:
                    new_columns.append(col)  # 如果没有匹配，保持不变
            assert sum(projection_cnt.values())==0
            # 替换列名
            df.columns = new_columns

        df.to_csv(csv_path, index=False, encoding='utf-8')  # utf-8-sig避免中文乱码



        print(f"已成功将 {file_path} 转为 {csv_path}")