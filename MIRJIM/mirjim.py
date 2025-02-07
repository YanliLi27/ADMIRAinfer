import numpy as np
from multi_deploy import MultiGUI4DL
import SimpleITK as sitk
from postprocess.get_head import return_head
from onnx_models.central_selector import central_selector


class GUI4MIRJIM(MultiGUI4DL):
    def __init__(self, model_path, name_str = "Img2Img-saved", inout_method = 'drag', geometry = "400x600+100+100", num_input = 2):
        super().__init__(model_path, name_str, inout_method, geometry, num_input)


    def _intensity_normalize(self, volume: np.array):
        min_value = volume.min()
        max_value = volume.max()
        if max_value > min_value:
            out = (volume - min_value) / (max_value - min_value + 1e-7)
        else:
            out = volume
        return out

    def _preprocess(self, data:str):
        """ 预处理图片，使其符合 ONNX 模型的输入要求 """
        data_list = []
        for i in range(self.num_input):
            file = sitk.ReadImage(data[i])
            file = sitk.GetArrayFromImage(file)
            # find the central slices
            if 'CORT1f' in data[i]:
                cs = central_selector(data[i])
                file = self._intensity_normalize(file[cs:cs+7])
            elif 'TRAT1f' in data[i]:
                if file.shape[0]//2 > 7:
                    center = file.shape[0]//2
                    s = slice(center-7, center+7, 2)
                    file = self._intensity_normalize(file[s])
            file = np.expand_dims(file, axis=0)  # [7, 512, 512] -> [1, 7, 512, 512]
            data_list.append(file)  # [] <-- np.array [1, 7, 512, 512]
        data = np.concatenate(data_list, axis=0)  #  [n, 7, 512, 512]
        data = np.expand_dims(data, axis=0)
        return data.astype(np.float32)

    def _postprocess(self, outputs):  # from score to RAMRIS
        # res_head:list = ['ID', 'ScanDatum', 'ID_Timepoint']  --> Not needed for visual outputs
        # TODO add the results
        outputs = np.maximum(outputs, 0)
        res_head:list = return_head("Wrist", "TSY")
        sorted_output = []
        for i in range(len(outputs)):
            sorted_output.append("{head}: {out:.1f}".format(head=res_head[i], out=outputs[i]))
        item = ", ".join(sorted_output)
        self.result_label.config(text=f"{item}")

    def _predict(self, path:list):
        """ 进行模型推理并返回分数 """
        self.path = path  # for post_process to understand what the site is
        data = self._preprocess(path)  # 读取数据也在其中

        inputs = {self.session.get_inputs()[0].name: data}
        outputs = self.session.run(None, inputs)  # --> np.array : RAMRIS outputs

        self._postprocess(outputs[0][0])

    

if __name__ == "__main__":
    site = 'Wrist'
    feature = 'TSY'
    app = GUI4MIRJIM(f"D:\\ESMIRAcode\\RAMRISinfer\\MIRJIM\\model_out\\{site}_{feature}_multiview_sumFalse_0.onnx", 
                 name_str="ONNXImgProcessing", inout_method="drag", num_input=2)
    app()