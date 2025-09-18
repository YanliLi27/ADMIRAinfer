import os
from pathlib import Path
import numpy as np
import onnxruntime as ort
from typing import Literal, Union, List

from admira_function import return_head, central_selector
from create_onnx import create_onnx_from_model  # necessary only when no onnx models exists, but pytorch model exists


class BaseADMIRA:
    def __init__(self, model_dir:str, site:Literal['Wrist', 'MCP', 'Foot'], feature:Literal['TSY', 'SYN', 'BME'],
                 model_type:List[str]):
        assert model_type in [['TRA'], ['COR'], ['TRA', 'COR']]
        model_path = self._obtain_model(model_dir, site, feature, model_type)
        if not Path.exists(model_path):
            raw_model_path:str = str(model_path).replace('onnx_model', 'raw_model')
            raw_model_path:str = raw_model_path.replace('.onnx', '.model')
            self._create_onnx(raw_model_path, model_path)
            assert Path.exists(model_path)
        self.session = self._load_model(model_path)
        self.site = site
        self.feature = feature
        self.model_type = model_type


    def _obtain_model(self, model_dir:str, site:Literal['Wrist', 'MCP', 'Foot'], 
                      feature:Literal['TSY', 'SYN', 'BME'], model_type:Union[List]) -> str:
        raise NotImplementedError("obtain model method requries to be customized")
    

    def _create_onnx(self, src, dst):
        raise NotImplementedError("create onnx function requries to be customized")
    

    def _load_model(self, model_path):
        """ 加载 Pytorch/ONNX 模型 """
        if os.path.exists(model_path):
            if ".onnx" in model_path: 
                return ort.InferenceSession(model_path, providers=['CPUExecutionProvider']) # providers=['CUDAExecutionProvider'])
            else: raise NotImplementedError("other types of deployment not implemented")
        raise AttributeError(f"{model_path} not exists!") 
    
    
    def _intensity_normalize(self, volume: np.array):
        min_value = volume.min()
        max_value = volume.max()
        if max_value > min_value:
            out = (volume - min_value) / (max_value - min_value + 1e-7)
        else:
            out = volume
        return out    


    def _preprocess(self, data:dict[str, np.ndarray]):
        """ preprocess the data to have the data shape as the onnx models required """
        raise NotImplementedError("has to be some pre-process for structured inputs")


    def _postprocess(self, outputs):  # from score to RAMRIS
        raise NotImplementedError("has to be some post-process for structured outputs")


    def predict(self, data:dict[str, np.ndarray]):
        """ model inference and obtain results """
        data = self._preprocess(data)  # 读取数据也在其中

        inputs = {self.session.get_inputs()[0].name: data}
        outputs = self.session.run(None, inputs)

        outputs = self._postprocess(outputs[0][0])

        return outputs
    


class TotalADMIRA(BaseADMIRA):
    def __init__(self, model_dir, site, feature, model_type):
        super().__init__(model_dir, site, feature, model_type)


    def _obtain_model(self, model_dir:str, site:Literal['Wrist', 'MCP', 'Foot'], 
                      feature:Literal['TSY', 'SYN', 'BME'], model_type:Union[List],
                      fold:int=0) -> str:
        mt = '2dirc' if len(model_type)>1 else f'{model_type[0]}'
        model_path:str = Path(model_dir)
        model_path:str = model_path / 'Total' / f'{feature}__{site}_{mt}_fold{fold}Sum.onnx'
        return model_path
    

    def _create_onnx(self, src, dst):
        create_onnx_from_model(src, dst, self.site, self.feature, self.model_type, True)


    def _preprocess(self, data:dict[str, np.ndarray]):
        """ preprocess the data to have the data shape as the onnx models required """
        data_list = []
        for key, file in data:
            # find the central slices
            if not file.shape[0]>6: raise ValueError('image has slice number less than 7')
            if key=='CORT1f':
                cs = central_selector(file)
                file = self._intensity_normalize(file[cs:cs+7])
            elif key=='TRAT1f':
                if file.shape[0]//2 > 7:
                    center = file.shape[0]//2
                    s = slice(center-7, center+7, 2)
                    file = self._intensity_normalize(file[s])
            file = np.expand_dims(file, axis=0)  # [7, 512, 512] -> [1, 7, 512, 512]
            data_list.append(file)  # [] <-- np.array [1, 7, 512, 512]
        data = np.concatenate(data_list, axis=0)  #  [n, 7, 512, 512]
        data = np.expand_dims(data, axis=0)
        return data.astype(np.float32)


    def _postprocess(self, outputs:np.ndarray) -> dict[str, float]:  # from score to RAMRIS
        outputs:np.ndarray = np.maximum(outputs, 0)  # regularize the outputs
        return {f'{self.site}_{self.feature}_inflammation': outputs}
        


class HighGranularityADMIRA(BaseADMIRA):
    def __init__(self, model_dir, site, feature, model_type) ->None:
        super().__init__(model_dir, site, feature, model_type)


    def _obtain_model(self, model_dir:str, site:Literal['Wrist', 'MCP', 'Foot'], 
                      feature:Literal['TSY', 'SYN', 'BME'], model_type:Union[List],
                      fold:int=0) -> str:
        mt = '2dirc' if len(model_type)>1 else f'{model_type[0]}'
        model_path:str = Path(model_dir)
        model_path:str = model_path / 'PerLocation' / f'{feature}__{site}_{mt}_fold{fold}.onnx'
        return model_path
    

    def _create_onnx(self, src, dst) -> None:
        create_onnx_from_model(src, dst, self.site, self.feature, self.model_type, False)


    def _preprocess(self, data:dict[str, np.ndarray]) -> np.ndarray:
        """ preprocess the data to have the data shape as the onnx models required """
        data_list = []
        for key, file in data:
            # find the central slices
            if not file.shape[0]>6: raise ValueError('image has slice number less than 7')
            if key=='CORT1f':
                cs = central_selector(file)
                file = self._intensity_normalize(file[cs:cs+7])
            elif key=='TRAT1f':
                if file.shape[0]//2 > 7:
                    center = file.shape[0]//2
                    s = slice(center-7, center+7, 2)
                    file = self._intensity_normalize(file[s])
            file = np.expand_dims(file, axis=0)  # [7, 512, 512] -> [1, 7, 512, 512]
            data_list.append(file)  # [] <-- np.array [1, 7, 512, 512]
        data = np.concatenate(data_list, axis=0)  #  [n, 7, 512, 512]
        data = np.expand_dims(data, axis=0)
        return data.astype(np.float32)


    def _postprocess(self, outputs) -> dict[str, float]:  # from score to RAMRIS
        outputs:np.ndarray = np.maximum(outputs, 0)  # regularize the outputs
        res_head:list = return_head(self.site, self.feature, False)  # obtain column names

        sorted_output:dict[str, float] = {}
        for i in range(len(outputs)):
            sorted_output[res_head[i]] = outputs[i]
        return sorted_output