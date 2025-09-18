import torch
import os
from .onnx_creation.csv3d import make_csv3dmodel
from typing import Literal, Optional, List, Any


def getmodel(site:List[Literal['Wrist','MCP','Foot']], 
             feature:List[Literal['TSY','SYN','BME']],
             model_type:Optional[List[str]]=['TRA', 'COR'],
             score_sum:bool=False):
    out_ch = 0
    if score_sum:
        out_ch = 1
    else:
        output_matrix = [[15, 15, 3, 10],[8, 8, 4, 8],[10, 10, 5, 10]]
        site_order = {'Wrist':0, 'MCP':1, 'Foot':2}
        bio_order = {'ERO':0, 'BME':1, 'SYN':2, 'TSY':3}
        out_ch = 0
        for si in site:
            for bi in feature:
                out_ch += output_matrix[site_order[si]][bio_order[bi]]
    model = make_csv3dmodel(img_2dsize=(7, 512, 512), inch=len(model_type), num_classes=2, num_features=out_ch, extension=0, 
                    groups=len(model_type), width=2, dsconv=False, attn_type='normal', patch_size=(2,2), 
                    mode_feature=True, dropout=False, init=False)
    return model  # ramris_Score, RApred


def getweight(src, model):
    if os.path.isfile(src):
        checkpoint = torch.load(src)
        model.load_state_dict(checkpoint)
        print('------------------------------------> Model load <------------------------------------')
        print(f'model loaded successfully: {src}')
    else:
        raise ValueError('model loading failed')
    return model


def init_torch_model(src:str,
                    site:Literal['Wrist','MCP','Foot'],
                    feature:Literal['TSY','SYN','BME'], 
                    model_tpye:List[str]=['TRA', 'COR'], 
                    score_sum:bool=False): 
    model = getmodel(site, feature, model_tpye, score_sum)
    model = getweight(src, model)
    model.eval() 
    return model 


def create_onnx_from_model(src, dst, 
                           site:Literal['Wrist','MCP','Foot'],
                           feature:Literal['TSY','SYN','BME'],
                           model_type:Optional[List[str]],
                           sumScore:bool
                           ):
    assert model_type in [['TRA'], ['COR'], ['TRA', 'COR']]
    input_range:int = 2 if len(model_type)>1 else 1
    x = torch.randn(1, input_range, 7, 512, 512)
    model = init_torch_model(src, [site], [feature], model_type, score_sum=sumScore) 

    dynamic_axes_23 = { 
        'input' : {0: 'batch', 3: "width", 4:"length"}, 
        'output' : {0: 'batch', 3: "width", 4:"length"} 
    } 

    with torch.no_grad(): 
        torch.onnx.export( 
            model, 
            x, 
            dst, 
            opset_version=11, 
            input_names=['input'], 
            output_names=['output'],
            dynamic_axes=dynamic_axes_23)