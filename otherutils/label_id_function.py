import os
import pandas as pd


def label_id(csvpath):
    campath = f'./output/ref/cam_use.xlsx'
    holdoutpath = f'./output/ref/holdout_names1.xlsx'
    csaintepath = f'R:\\AIMIRA\\AIMIRA_Scores\\PatientsFromCSA.xlsx'

    df = pd.read_csv(csvpath)
    cam_ids = pd.read_excel(campath)['ID'].astype(str)
    holdout_ids = pd.read_excel(holdoutpath)['ID'].astype(str)
    csainte = pd.read_excel(csaintepath)['CSA ID'].astype(str)
    csainte = [item.replace('X', '') for item in csainte]

    # 确保主表中的 ID 也是字符串以便匹配
    df['ID'] = df['ID'].astype(str)
    df['cam_use'] = df['ID'].isin(cam_ids).astype(int)
    # print(list(set(cam_ids) - set(df['ID'])))
    df['holdout'] = df['ID'].isin(holdout_ids).astype(int)
    df['inTE'] = df['ID'].isin(csainte).astype(int)
    print(f"cam_use=1 的数量: {df['cam_use'].sum()}")
    print(f"holdout=1 的数量: {df['holdout'].sum()}")
    print(f"te=1 的数量: {df['inTE'].sum()}")
    # 保存结果（可选）
    df.to_csv(csvpath.replace('.csv', '_tagged.csv'), index=False)