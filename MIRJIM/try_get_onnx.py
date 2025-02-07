import torch
from torch import nn


class SRNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        self.conv1 = nn.Conv2d(3,64,kernel_size=9,padding=4) 
        self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0) 
        self.conv3 = nn.Conv2d(32,3,kernel_size=5,padding=2) 
        self.relu = nn.ReLU()
    def forward(self, x): 
        x = self.img_upsampler(x) 
        out = self.relu(self.conv1(x)) 
        out = self.relu(self.conv2(out)) 
        out = self.conv3(out) 
        return out

def init_torch_model(): 
    torch_model = SRNet(upscale_factor=3) 
    state_dict = torch.load(r'C:\Yanli\MyCode\RL\DeployExample\srcnn.pth')['state_dict'] 
 
    # Adapt the checkpoint 
    for old_key in list(state_dict.keys()): 
        new_key = '.'.join(old_key.split('.')[1:]) 
        state_dict[new_key] = state_dict.pop(old_key) 
 
    torch_model.load_state_dict(state_dict) 
    torch_model.eval() 
    return torch_model 

x = torch.randn(1, 3, 256, 256) 
model = init_torch_model() 

dynamic_axes_23 = { 
    'input' : {0: 'batch', 2: "width", 3:"depth"}, 
    'output' : {0: 'batch', 2: "width", 3:"depth"} 
} 

with torch.no_grad(): 
    torch.onnx.export( 
        model, 
        x, 
        r"C:\Yanli\MyCode\RL\MIRJIM\srcnn.onnx", 
        opset_version=11, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes=dynamic_axes_23)
    # pytorch是动态的，但是ONNX等类型的是先编译然后再执行，所以说会需要给入一个输入，然后走一轮看看整个网络到底是怎么跑的
    