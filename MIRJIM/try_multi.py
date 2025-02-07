import numpy as np
from tkinterdnd2 import TkinterDnD
from PIL import Image, ImageTk
from multi_deploy import MultiGUI4DL
import SimpleITK as sitk
from models.central_selector import central_selector
import cv2


class Multi4IMG2IMG(MultiGUI4DL):
    def __init__(self, model_path, name_str = "Img2Img-saved", inout_method = 'drag', geometry = "400x600+100+100", num_input = 2):
        super().__init__(model_path, name_str, inout_method, geometry, num_input)


    def _preprocess(self, data:list):
        """ 预处理图片，使其符合 ONNX 模型的输入要求 """
        data_list = []
        for img in data:
            image = cv2.imread(img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))  # 假设模型输入尺寸为 224x224
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            image = np.expand_dims(image, axis=0)  # 增加 batch 维度
            data_list.append(image)
        data_array:np.array = np.concatenate(data_list, axis=0)
        return data_array

    def _postprocess(self, data):
        data = np.transpose(data, (1, 2, 0))  # CHW -> HWC
        data = (data * 255).astype(np.uint8)  # 归一化到 0-255
        data = Image.fromarray(data)
        photo_output = ImageTk.PhotoImage(data)
        self.result_label.config(image=photo_output)
        self.result_label.image = photo_output
    
    def _predict(self, path):
        """ 进行模型推理并返回分数 """
        if len(self.name_str)>0:
            save = f"{self.name_str}_" + path.split(r"/")[-1]
        else: save = None
        data = self._preprocess(path)  # 读取数据也在其中 [B, C, H, W]

        inputs = {self.session.get_inputs()[0].name: data}
        outputs = self.session.run(save, inputs)

        outputs = self._postprocess(outputs[0][0])  # [B, C, H, W] -> []
        self._output_standard(outputs)


if __name__ == "__main__":
    
    app = Multi4IMG2IMG("C:\\Yanli\\MyCode\\RL\\MIRJIM\\srcnn.onnx", 
                 name_str="ONNXImgProcessing", inout_method="drag")
    app()