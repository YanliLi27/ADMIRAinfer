import torch
import os
from torch import nn
from typing import Literal, Optional, List
from onnx_models.get_model import getmodel
from onnx_models.get_weight import getweight


def init_torch_model(site:Literal['Wrist','MCP','Foot'],
                    feature:Literal['TSY','SYN','BME'], 
                    view:Optional[List[str]]=['TRA', 'COR'], 
                    score_sum:bool=False): 
    model = getmodel(site, feature, view, score_sum)
    model = getweight(model, site, feature, score_sum, view, order=0)
    model.eval() 
    return model 


def create_onnx(site:Literal['Wrist','MCP','Foot'],
                feature:Literal['TSY','SYN','BME']):
    x = torch.randn(1, 2, 7, 512, 512) 
    model = init_torch_model([site], [feature]) 

    dynamic_axes_23 = { 
        'input' : {0: 'batch', 3: "width", 4:"length"}, 
        'output' : {0: 'batch', 3: "width", 4:"length"} 
    } 

    path = os.path.join("D:\\ESMIRAcode\\RAMRISinfer\\MIRJIM\\model_out", f"{site}_{feature}_multiview_sumFalse_0.onnx")
    with torch.no_grad(): 
        torch.onnx.export( 
            model, 
            x, 
            path, 
            opset_version=11, 
            input_names=['input'], 
            output_names=['output'],
            dynamic_axes=dynamic_axes_23)
        # pytorch是动态的，但是ONNX等类型的是先编译然后再执行，
        # 所以说会需要给入一个输入，然后走一轮看看整个网络到底是怎么跑的

if __name__=="__main__":
    create_onnx('Wrist', 'TSY')
    create_onnx('MCP', 'TSY')
    create_onnx('Foot', 'TSY')
        