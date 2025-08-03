
import os
import pandas as pd


if __name__=='__main__':
    csvpath = f'./output/all/3sitemerged_ALL_sumFalse.csv'
    campath = f'./output/all/cam_use.xlsx'
    holdoutpath = f'./output/all/holdout_names1.xlsx'

    df = pd.read_csv(csvpath)
    cam_ids = pd.read_excel(campath)['ID'].astype(str)
    holdout_ids = pd.read_excel(holdoutpath)['ID'].astype(str)

    # 确保主表中的 ID 也是字符串以便匹配
    df['ID'] = df['ID'].astype(str)
    df['cam_use'] = df['ID'].isin(cam_ids).astype(int)
    df['holdout'] = df['ID'].isin(holdout_ids).astype(int)
    print(f"cam_use=1 的数量: {df['cam_use'].sum()}")
    print(f"holdout=1 的数量: {df['holdout'].sum()}")
    # 保存结果（可选）
    df.to_csv('./output/all/3sitemerged_ALL_sumFalse_tagged.csv', index=False)