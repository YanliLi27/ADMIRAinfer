import pandas as pd
import os



def merge_from_two_csv(score_csv:str=r'D:\ESMIRAcode\ADMIRAinfer\output\outside\holdout_score\3sitemerged_ALL_sumTrue.csv', 
                       std_csv:str=r'D:\ESMIRAcode\ADMIRAinfer\output\outside\holdout_std\3sitemerged_ALL_sumTrue.csv',
                       output:str=r'D:\ESMIRAcode\ADMIRAinfer\output\outside\3sitemerged_ALL_sumTrue.csv'):
    df_a = pd.read_csv(score_csv)
    df_b = pd.read_csv(std_csv)

    # 设置索引
    df_a_indexed = df_a.set_index(['ID', 'ScanDatum', 'ID_Timepoint'])
    df_b_indexed = df_b.set_index(['ID', 'ScanDatum', 'ID_Timepoint'])

    # 获取所有带'_std'的列名
    std_columns = [col for col in df_b.columns if '_std' in col]

    # 只选择std列进行更新
    df_b_std_only = df_b_indexed[std_columns]

    # 更新A的对应列
    df_a_indexed.update(df_b_std_only)

    # 重置索引
    df_result = df_a_indexed.reset_index()

    # 保存结果
    df_result.to_csv(output, index=False)



if __name__=="__main__":
    score_sum_list = [True, False]
    for ss in score_sum_list:
        score_path:str = f'D:\\ESMIRAcode\\ADMIRAinfer\\output\\outside\\holdout_score\\3sitemerged_ALL_sum{ss}.csv'
        std_path:str = f'D:\\ESMIRAcode\\ADMIRAinfer\\output\\outside\\holdout_std\\3sitemerged_ALL_sum{ss}.csv'
        output:str = f'D:\\ESMIRAcode\\ADMIRAinfer\\output\\outside\\3sitemerged_ALL_sum{ss}.csv'
        merge_from_two_csv(score_path, std_path, output)