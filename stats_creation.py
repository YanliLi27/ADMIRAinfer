from typing import Literal, Optional, List
import pandas as pd
import torch
from datasets.get_data import getdata
from models.get_model import getmodel
from trained_weights.get_weight import getweight
from utils.get_head import return_head, return_head_gt
from torch.utils.data import DataLoader
from tqdm import tqdm

# 从 E:\ADMIRA_models\weights\sumFalse、True 获取 {BIO}__{SITE}_2dirc_fold{FOLD}.model的权重 BME__Foot_2dirc_fold3.model

# 从 getdata获得对应的fold的 monitoring 数据
# 从 E:\ADMIRA_models\split\sumFalse、True 获取monitoring的数据分割：
# 用pkl_reader把它们转化为csv数据，然后用和此处的dataset相同的形式进行保存
# 【un22_EAC_CSA_ATL__{SITE}_2dirc_1.pkl】  
# 里面存的是 [5split  *[id[path1:cs, path2:cs, path3:cs, ...]的list
# 【un22_EAC_CSA_ATL__{SITE}_{BIO}_2reader_1__sc.pkl 
# 里面存的是对应BIO的 [5split  *[id[site1_array, site2_array], id[[site1_array, site2_array]]

# 每次把对应的数据和对应的模型进行一波inference，然后存到一个csv当中，首先进行一个site下的全split合并
# 然后合并三个site到一个csv，最后通过桌面的hold的标记增加一列标注他们的类别
# 然后根据桌面的CAM.xlsx对之前的做了obeserver study的进行增列标注

# 同样的用训练好的模型对TE进行一个inference - 使用main.py，保存成类似的情况下，然后类似合并