from typing import Literal, List, Optional
import os
import torch


def getweight(model, site:Literal['Wrist','MCP','Foot'],
            feature:Literal['TSY','SYN','BME'], 
            sumscore:bool=True,
            view:Optional[List[str]]=['TRA', 'COR'], 
            order:Optional[int]=0):
    if len(feature)==1 and not view: view = ['COR'] if  feature[0] in ['SYN', 'BME'] else ['TRA']
    if len(view)<2:
        si = '_'.join(site)
        fa = '_'.join(feature)
        path = f'D:\\ESMIRAcode\\RAMRISinfer\\trained_weights\\{si}_{fa}_{view[0]}_sum{sumscore}_{order}.model'
    else:
        si = '_'.join(site)
        fa = '_'.join(feature)
        path = f'D:\\ESMIRAcode\\RAMRISinfer\\trained_weights\\{si}_{fa}_multiview_sum{sumscore}_{order}.model'
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        print('------------------------------------> Model load <------------------------------------')
        print(f'model loaded successfully: {path}')
    else:
        raise ValueError('model loading failed')
    return model