# 获得可能的模型从trained_weights那边
from typing import Literal,  Optional, List
from .clip_model import ModelClip
from .convsharevit import make_csvmodel
from .csv3d import make_csv3dmodel

def getmodel(site:Literal['Wrist','MCP','Foot'], 
             feature:Literal['TSY','SYN','BME'],
             view:Optional[List[str]]=['TRA', 'COR'],
             score_sum:bool=False):
    out_ch = 0
    if score_sum:
        out_ch = 1
    else:
        output_matrix = [[15, 15, 3, 10],[8, 8, 4, 8],[10, 10, 5, 10]]
        site_order = {'Wrist':0, 'MCP':1, 'Foot':2}
        bio_order = {'ERO':0, 'BME':1, 'SYN':2, 'TSY':3}
        out_ch = output_matrix[site_order[site]][bio_order[feature]]
    model = make_csv3dmodel(img_2dsize=(7, 512, 512), inch=len(view), num_classes=2, num_features=out_ch, extension=0, 
                    groups=len(view), width=2, dsconv=False, attn_type='normal', patch_size=(2,2), 
                    mode_feature=True, dropout=False, init=False)
    return model