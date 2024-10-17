from typing import Literal
import os
import torch


def getweight(model, site:Literal['Wrist','MCP','Foot'],
            feature:Literal['TSY','SYN','BME'], order:int=0):
    view = 'COR' if feature in ['SYN', 'BME'] else 'TRA'
    path = f'./trained_weights/{site}_{feature}_{view}_{order}.model'
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        print('------------------------------------> Model load <------------------------------------')
        print(f'model loaded successfully: {path}')
    else:
        raise ValueError('model loading failed')
    return model