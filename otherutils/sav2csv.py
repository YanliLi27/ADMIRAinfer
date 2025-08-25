import pandas as pd
import pyreadstat

# 读取sav文件
path_lists = [r'R:\ESMIRA\ESMIRA_Scores\SPSS data\5. CSA_T1_MRI_scores_SPSS.sav',
              r'R:\ESMIRA\ESMIRA_Scores\SPSS data\6. CSA_T2_MRI_scores_SPSS.sav',
            r'R:\ESMIRA\ESMIRA_Scores\SPSS data\7. CSA_T4 MRI_scores_SPSS.sav',  # CSANUMM
            r'R:\ESMIRA\ESMIRA_Scores\SPSS data\1. EAC baseline.sav', 
            r'R:\ESMIRA\ESMIRA_Scores\SPSS data\2. EAC longitudinal.sav',
            r'R:\ESMIRA\ESMIRA_Scores\SPSS data\3. Atlas.sav']
for file_path in path_lists:

    df, meta = pyreadstat.read_sav(file_path)

    # 查看前几行数据
    print(df.head())

    # 转换为CSV
    csv_path = file_path.replace('.sav', '.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')  # utf-8-sig避免中文乱码

    print(f"已成功将 {file_path} 转为 {csv_path}")