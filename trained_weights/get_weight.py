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


def getweight_outside(model, 
            saving_dir:str=r'E:\ADMIRA_models\weights',
            site:Literal['Wrist','MCP','Foot']='Wrist',
            feature:Literal['TSY','SYN','BME']='TSY', 
            sumscore:bool=True,
            view:Optional[List[str]]=['TRA', 'COR'], 
            order:Optional[int]=0):
    # E:\ADMIRA_models\weights\sumFalse、True 获取 {BIO}__{SITE}_2dirc_fold{FOLD}.model
    if not view: view = ['COR'] if feature in ['SYN', 'BME'] else ['TRA']
    ss = f'sum{sumscore}'
    if len(view)<2:
        path = f'{feature}__{site}_{view}_fold{order}.model'
    else:
        path = f'{feature}__{site}_2dirc_fold{order}.model'
    path = os.path.join(saving_dir, ss, path)
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        print('------------------------------------> Model load <------------------------------------')
        print(f'model loaded successfully: {path}')
    else:
        raise ValueError('model loading failed')
    return model