import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import filedialog, Label
import onnxruntime as ort
from typing import Literal
import os


class MultiGUI4DL:
    def __init__(self, model_path:str, name_str:str="Img2Img-saved", 
                 inout_method:Literal['drag', 'select']='drag',
                 geometry:str="400x600+100+100",
                 num_input:int=2):
        self.root = TkinterDnD.Tk()
        self.root.title(f"DL model for {name_str}")
        self.root.geometry(geometry) 
        # "<Width>x<Height>+<x_pos>+<y_pos>"

        # load model
        self.session = self._load_model(model_path)
        # entry for GUI
        self.num_input:int = num_input
        self.entry = {}  # int->GUI.entry
        self.path = {}  # int->GUI.path
        self.label = {}  # int->GUI.label

        self.method = inout_method
        self.name_str:str = name_str.split('-')[0] if "-saved" in name_str else ""
        self._create_widgets() # 创建GUI组件

    def _select_file(self, idx:int):
        file_path = filedialog.askopenfilename(title=f"choose file{idx}",
                                               filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.nii;*.nii.gz;*.mha")])
        if file_path: 
            self.label[idx].config(text=f"{file_path}")
            self.path[idx] = file_path
        else: raise TypeError("not found available files")
    
    def _on_drop(self, event, entry):
        # deal with the drag event
        entry.delete(0, tk.END)
        entry.insert(0, event.data.strip('{}'))

    def _run_model(self):
        file_paths:list = []
        if self.method=="drag":
            for i in range(self.num_input):
                file_paths.append(self.entry[i].get())
        elif self.method=="select":
            file_paths:list = list(self.path.values())
        # the calculation process:
        self._predict(file_paths)
          
    def _create_widgets(self):
        """ 创建 GUI 组件 """
        for i in range(self.num_input):
            self.label[i] = tk.Label(self.root, text=f"--> Drag Input {i} here <--")
            self.label[i].pack(pady=10)

            if self.method == "drag":
                self.entry[i] = tk.Entry(self.root, width=50)
                self.entry[i].pack(pady=10)
                self.entry[i].drop_target_register(DND_FILES)
                self.entry[i].dnd_bind('<<Drop>>', lambda event, idx=i: self._on_drop(event, self.entry[idx]))
            elif self.method == "select":
                self.entry[i] = tk.Button(self.root, text=f"choose figure {i}", command=lambda idx=i: self._select_file(idx))
                self.entry[i].pack()
            else: raise NotImplementedError("other inputs require further development")

        # create run
        self.run_button = tk.Button(self.root, text="--> Run <--", command=self._run_model)
        self.run_button.pack(pady=20)

        # show results
        self.result_label = Label(self.root, text="Outputs")
        self.result_label.pack()
        
    def _load_model(self, model_path):
        """ 加载 Pytorch/ONNX 模型 """
        if os.path.exists(model_path):
            if ".onnx" in model_path: return ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
            else: raise NotImplementedError("other types of deployment not implemented")
        raise AttributeError(f"{model_path} not exists!")
    
    def _preprocess(self, data):
        raise NotImplementedError("has to be some pre-process, including loading")
    
    def _postprocess(self, data):
        raise NotImplementedError("has to be some post-process")    

    def _predict(self, path:list):
        """ 进行模型推理并返回分数 """
        data = self._preprocess(path)  # 读取数据也在其中

        inputs = {self.session.get_inputs()[0].name: data}
        outputs = self.session.run(None, inputs)

        outputs = self._postprocess(outputs[0][0])
        self._output_standard(outputs)
        
    def __call__(self):
        self.root.mainloop()

    def __del__(self):
        print("Session ended!")


if __name__ == "__main__":
    app = MultiGUI4DL("C:\\Yanli\\MyCode\\RL\\GUI_ONNX_Deploy\\srcnn.onnx", 
                 name_str="ONNXImgProcessing", inout_method="drag")
    app()
