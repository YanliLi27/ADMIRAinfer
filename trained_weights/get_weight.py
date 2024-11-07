from typing import Literal, List, Optional
import os
import torch


def getweight(model, site:Literal['Wrist','MCP','Foot'],
            feature:Literal['TSY','SYN','BME'], 
            sumscore:bool=True,
            view:Optional[List[str]]=['TRA', 'COR'], 
            order:Optional[int]=0):
    if not view: view = ['COR'] if feature in ['SYN', 'BME'] else ['TRA']
    if len(view)<2:
        path = f'./trained_weights/{site}_{feature}_{view[0]}_sum{sumscore}_{order}.model'
    else:
        path = f'./trained_weights/{site}_{feature}_multiview_sum{sumscore}_{order}.model'
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        print('------------------------------------> Model load <------------------------------------')
        print(f'model loaded successfully: {path}')
    else:
        raise ValueError('model loading failed')
    return model