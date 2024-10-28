from seg_components import UNet
import os
import torch


def model_seg_find(in_channel, n_classes, width, site, fold_order:int=0):
    model_seg = UNet(n_channels=in_channel, n_classes=n_classes, width=width, softmax_flag=False, bilinear=False)

    dim_extend = True if 'cor' in site else False
    output_name = f'D:\\Seg\\AggregatedCAM\\SegComponents\\output\\models\\{site}_munet_ce_dimex{dim_extend}_w{width}_{fold_order}_seg.model'
    if os.path.isfile(output_name):
        checkpoint = torch.load(output_name)
        model_seg.load_state_dict(checkpoint)
        print(f'load model: {output_name}')
        return model_seg
    else:
        raise ValueError(f'model weights not exist: {output_name}')