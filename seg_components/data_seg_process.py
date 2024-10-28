import torch

def tra_seg(x:torch.Tensor, slices:int=7)->torch.Tensor:
    x = x[:, :slices, :]  # [batch, 5, L, W]
    return x  # [batch, slices_tra, L, W]


def cor_seg(x:torch.Tensor, slices:int=7)->torch.Tensor:
    x = x[:, -slices:, :]  # [batch, 6-10, L, W]
    B, C, L, W = x.shape
    x_p = torch.zeros((B,C-2,3,L,W), dtype=x.dtype)
    for i in range(0, C-2, 1):
        x_p[:, i, :] = x[:, i:i+3, :]  # [batch, i, 3, L, W] <-- [batchc, 3, L, W]
        # [B, i, 3, L, W] -> [B, 3, L, W] = [B, 3, L, W] -> [B, 3, L, W]
    return x_p  # [batch, slices-2, 3, L, W]


def cor_seg3d(x:torch.Tensor)->torch.Tensor:
    # x_cor [batch, slice, L, W]
    B, C, L, W = x.shape
    x_p = torch.zeros((B,C-2,3,L,W), dtype=x.dtype)
    for i in range(0, C-2, 1):
        x_p[:, i, :] = x[:, i:i+3, :]  # [batch, i, 3, L, W] <-- [batchc, 3, L, W]
        # [B, i, 3, L, W] -> [B, 3, L, W] = [B, 3, L, W] -> [B, 3, L, W]
    return x_p  # [batch, slices-2, 3, L, W]


def data_process(x:torch.Tensor, num_input:int=2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: 2D [batch, channel, L, W], 3D [batch, channel, slice, L, W]
    # x [batch, channel(N*5/7), L, W]
    # input: x,y (from the dataloader-> for x,y in dataloader)
    # return the datasets for class_model, tra_seg_model, cor_seg_model, with preprocessing
    if len(x.shape)==4:
        _, C, _, _ = x.shape
        slices = C//num_input
        tra_seg_x = tra_seg(x, slices=slices)
        cor_seg_x = cor_seg(x, slices=slices)
    elif len(x.shape)==5:
        tra_seg_x = x[:, 0, :]  # [batch, slice, L, W]
        cor_seg_x = cor_seg3d(x[:, 1, :])  # [batch, slices-2, 3, L, W]
    return x, tra_seg_x, cor_seg_x

