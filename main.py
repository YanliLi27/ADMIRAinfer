# model读取可以采用原本的读取改动一下，把训练好的结果放在trained_weights
# 数据读取要采用新的读取：
# 首先重新获得需要计算的central slice，是否类似ACAM一样重新保存？
# 也可以直接on-fly进行操作，因为只需要计算一次
# 分别计算Wrist,MCP,MTP; TSY,SYN,BME的结果，每个分别保存到outputs中的一个:
#  f'{Site}_{Feature}_{Group}.csv'
# 最后用一个pandas merge在另一个文件里面把所有的结果merge起来，根据ID, time_point来计算
# 本程序只需要循环Site和Feature即可
from typing import Literal, Optional, List
import pandas as pd
import torch
from .datasets.get_data import getdata
from .models.get_model import getmodel
from .trained_weights.get_weight import getweight
from .utils.get_head import return_head, return_head_gt
from .utils.get_path import getpath
from torch.utils.data import DataLoader



def main_process(task:Literal['CSA', 'TE'], site:Literal['Wrist','MCP','Foot'],
                 feature:Literal['TSY','SYN','BME'], 
                 view:Optional[List[str]]=['TRA', 'COR'],
                 order:int=0, score_sum:bool=False, filt:Optional[list]=None):
    # 模型本身和权重都需要site feature
    model = getmodel(site, feature, view, score_sum)  # DONE!
    model = getweight(model, site, feature, view, order)  # DONE!
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # 数据读取需要task, site, feature
    # 数据返回应该是 img, scores, path (id, date)
    data, maxidx = getdata(task, site, feature, view, filt, score_sum=False, path_flag=True)
    # CSA那个需要返回所有选择的CSA的列表并进行On-fly选中间的切片
    # TE那个需要返回所有的Timepoint的列表并进行On-fly选中间的切片
    # x,y,z: img, scores, path
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1)

    # create results dict for csv file
    res_head= ['ID', 'ScanDatum', 'ID_Timepoint']
    res_head.extend(return_head(site, feature))
    res_head.extend(return_head_gt(site, feature))
    df = pd.DataFrame(index=range(maxidx), columns=res_head)
    idx = 0
    for x, y, z in dataloader:
        x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # Tensor
        pred:torch.Tensor = model(x)  # [B, num_scores] Tensor
        for i in range(x.shape[0]):
            pid, ptp = getpath(z[i].cpu().numpy())    # getpath Done!
            row = [pid, ptp, f'{pid}_{ptp}']
            row.extend(pred[i].cpu().numpy())
            row.extend(y[i].cpu().numpy())
            df.loc[idx] = row
            idx += 1
        # 用pd.concat([df, new_row], ignore_index=True)来添加新的一行数据
    df.to_csv(f'./outputs/{site}_{feature}_{task}.csv')

    # inference:
    # 直接for x,y,z in Dataloader():
    # pred = model(x)
    # 根据z划分出ID和time_point(只需要有时间就可以了，
    # 再存一个ID_TP: CSA001_20140901用来合并
    # 存到一个csv里面

if __name__=='__main__':
    for site in ['Wrist', 'MCP', 'Foot']:
        for feature in ['TSY','SYN','BME']:
            main_process('CSA', site, feature, view=['TRA', 'COR'], order=0, score_sum=False, filt=None)
