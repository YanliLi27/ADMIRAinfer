# Automatic joint inflammation estimation based on regression neural networks (06/2025 updated)
</p>
<!--<h1 align="center"><b>Quantus</b></h1>-->
<h3 align="center"><b>A pytorch repository to assess joint inflammation based on MRIs using DL models</b></h3>
<p align="center">
  PyTorch
<p align="center">
  > Input support: 2D, 3D MRI images (wrist, MCP, MTP joints), Outputs: inflammation scores for (teno-)synovitis and bone marrow edema (BME), 
  > Other Outputs: heatmaps (saliency maps), population-level analysis on the contribution of each anatomical structure to outputs.

<!--[Shortcut to arxiv draft](https://arxiv.org/abs/2407.01142) -->


## Table of contents
* [Library overview](#library-overview)
* [Getting started](#getting-started)
* [GUI interface](#gui-interface)
* [Agggregation, combined with segmentation](#Aggregation)


## Library overview
<details>
<summary><b><big>Purpose of this library</big></b></summary>
The purpose of this library is to provide a overview of model architecture, configurations and easy-use GUI interface of our Automatic DL-based MRI analysis of Inflammatory signs in RA (ADMIRA) system for inflammation assessment. It provides details on the training and validation of the study to serve as a reference for future study on similar topics, and provides a basic application of image-in, scores-out automatic system for MRI inflammatio assessment to show the potential ability of such systems in this field.
</details>


<details>
<summary><b><big>Code structure</big></b></summary>
<li>The code structure of this library includes:</li>

> Model architectures
  <details>
  <summary><b><big>models</big></b></summary>
  the folder `models` contains the core functionality of this method with the following structure:
    <li>csv3d.py: the 3D models for inflammation assessment.</li>
    <li>convsharevit.py: the 2D models for inflammation assessment.</li>
    <li>clip_model: a 2D models for inflammation assessment based on pure Convolution neural networks.</li>
    <li>others: supporting files </li>
  </details>


> GUI interface:
  <details>
  <summary><b><big>MIRJIM</big></b></summary>
  The folder contains the codes used for building a onnx- and tkinter- based simple GUI application of the ADMIRA, used for inference runtime evaluation - ~1 second/case based on CPU running the trained models.
  
  > run the `mirjim.py` to launch the application.
  </details>

  
> Other utils:
  <details>
  <summary><b><big>seg_component</big></b></summary>
  The folder contains the code used for the segmentaion of MRIs, for aggregation of the generated saliency maps.
  </details>

</details>


## Getting started with python scripts
The following sections will give a short introduction to how to get started with the automatic inflammation assessment - one by run python scripts and one by GUI interface. This section is for python scripts as example given in `main.py`.

The required materials needed:
> A model (models in the `model`), inputs (MRI images as provided by `./datasets/get_data.py`)

<details>
<summary><b><big>Step 1. Set up data and model</big></b></summary>

The first step is to have the data and model, here we take a trained model in `./trained_weights/Wrist_BME_COR_sumTrue_0.model` as an examples.

```python
import torch
from datasets.get_data import getdata
from models.get_model import getmodel
from trained_weights.get_weight import getweight
from torch.utils.data import DataLoader

# get model
model = getmodel(site, feature, view, score_sum)
# get trained weights
model = getweight(model, site, feature, score_sum, view, order)
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval()

# get dataset from predefined functions
data, maxidx = getdata(task, site, feature, view, filt, score_sum, path_flag=True)
# set dataloader to feed model
dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1)
```
</details>

<details>
<summary><b><big>Step 2. Create a storage for the output with label</big></b></summary>

The second step is to create a list to store the output.

```python
from utils.get_head import return_head, return_head_gt

res_head= ['ID', 'ScanDatum', 'ID_Timepoint']
if not score_sum:
    res_head.extend(return_head(site, feature))
    res_head.extend(return_head_gt(site, feature))
else:
    res_head.extend(['sums', 'sums_gt'])
df = pd.DataFrame(index=range(maxidx), columns=res_head)
```
</details>


<details>
<summary><b><big>Step 3. run and get results</big></b></summary>

```python
for x, y, z in tqdm(dataloader):
    x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Tensor
    with torch.no_grad():
        pred:torch.Tensor = model(x)  # [B, num_scores] Tensor
        for i in range(x.shape[0]):
            pid, ptp = z[i].split('_')  # getpath Done!
            row = [pid, ptp, f'{pid}_{ptp}']
            row.extend(pred[i].cpu().numpy())
            row.extend(y[i].cpu().numpy())
            df.loc[idx] = row
            idx += 1
    # 用pd.concat([df, new_row], ignore_index=True)来添加新的一行数据
df.to_csv(f'./output/{site}_{feature}_{task}_sum{score_sum}.csv')
```

Or you can also feed with single input or unlabelled inputs:

```python
data = torch.from_numpy(np.load('data_source'))
data = torch.unsqueeze(data, dim=0)  # make it 4 dimensional

output = model(x)
df.loc[idx] = output

df.to_csv(f'./output/{site}_{feature}_{task}_sum{score_sum}.csv')
```
</details>


## Getting started with GUI interface
The following contents explains how to obtain the same outputs as python scripts with a GUI interface.

<details>
<summary><b><big>Step -1. run GUI interface</big></b></summary>

> Run `MIRJIM/mirjim.py` with customized hyperparameters:
```python
site = 'anatomical region'
feature = 'inflammation sign'
app = GUI4MIRJIM(f"D:\\ESMIRAcode\\RAMRISinfer\\MIRJIM\\model_out\\{site}_{feature}_multiview_sumFalse_0.onnx", 
                name_str="ONNXImgProcessing", inout_method="drag", num_input=2)
app()
```
</details>

> Then drag MRIs and put them into the boxes in the interface.

> Click `run`.


## Citation
> for cite this repositry, please cite: Integrated feature analysis for deep learning interpretation and class activation maps, arxiv.org/abs/2407.01142


