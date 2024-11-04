import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, binary_closing, binary_dilation


def overlay_postprocess(x:np.array):
    # x [batch, slice, 33, L, W]
    B, S, C, L, W = x.shape
    x_bones = np.max(x[:, :, 19:, :], axis=2)  # [B, S, C(14), L, W] --> [B,S,L,W]
    x_nonebone = np.ones((B,S,L,W)) - x_bones  # [B, S, L, W]
    for c in range(19):
        x[:, :, c] = np.minimum(x[:, :, c], x_nonebone)
    return x # [batch, 33, L, W]


def tra_slicemorph(x:np.array) ->np.array:
    # x [batch, slice, 33, L, W]
    B, S, C, _, _ = x.shape
    for m in range(B):
        for s in range(S):
            for n in range(C):
                if n <18 and n>=4:  # tendon + vessels
                    x[m, s, n] = binary_opening(x[m, s, n], iterations=2)  # closing 1 time to avoid some strange dots outside
                    x[m, s, n] = binary_dilation(x[m, s, n], iterations=20)  # 向外扩张5次， a tendon is nearly 10*10 in average in TRA scans
                elif n >=18:  # bones
                    x[m, s, n] = binary_closing(x[m, s, n], iterations=10)  # fix the holes and BMEs, and avoid some strange dots outside, as bones are larger
    return overlay_postprocess(x)  # x [batch, slice, 33, L, W]


def cor_slicemorph(x:np.array) ->np.array:
    # x [batch, slice, 33, L, W]
    B, S, C, _, _ = x.shape
    for m in range(B):
        for s in range(S):
            for n in range(C):
                if n>=4:  # tendon + vessels
                    x[m, s, n] = binary_closing(x[m, s, n], iterations=15)  # 向外扩张5次， a tendon is nearly 10*10 in average in TRA scans
    return x  # x [batch, slice, 33, L, W]


def merge_seg(x:np.array, tra:bool, merge:bool=True) ->np.array:
    # x [batch, slice, 33, L, W]
    assert len(x.shape)==5
    x = tra_slicemorph(x) if tra else cor_slicemorph(x)  # x [batch, slice, 33, L, W]
    x = np.max(x, axis=1) if merge else np.transpose(x, (0, 2, 1, 3, 4)) # x [batch, 33, L, W], # x [batch, 33, D, L, W]
    return x.astype(np.float32)  # [batch, 33, L, W]/ x [batch, 33, D, L, W]


def post_seg(x:np.array):
    # [batch, slice, 33, L, W]
    B, S, C, L, W = x.shape
    reseg = np.zeros((B,S,C,L,W), dtype=np.int16)
    for b in range(B):
        for s in range(S):
            mask = np.argmax(x[b, s, :], axis=0)
             # [L, W]
            masks = [(mask == v) for v in list(range(C))]  # [L,W]
            reseg[b,s] = np.stack(masks, axis=0).astype('int')  # [C,L,W]
    return reseg  # [batch, slice, 33, L, W] binarized


def segtra(x:torch.Tensor, model, merge:bool=True, raw:bool=False) ->np.array:
    # x [batch, slice, L, W]
    seg_list = []
    for i in range(x.shape[1]):
        seg_list.append(model(x[:, i, :].unsqueeze(dim=1)).cpu().data.numpy())  # [batch, 33, L, W]  -> [slice, batch, 33, L, W]
    seg_list = np.transpose(np.asarray(seg_list), (1, 0, 2, 3, 4))  # [slice, batch, 33, L, W] -> [batch, slice, 33, L, W]
    seg_list = post_seg(seg_list)
        # for seg in seg_list:
        #     for sli in seg:
        #         for c in sli:
        #             plt.imshow(c)
        #             plt.show()
        #             pass
        # [batch, slice, 33, L, W] binarized
    if raw:
        return np.transpose(np.asarray(seg_list), (0, 2, 1, 3, 4))   # [batch, slice, 33, L, W]-> [batch, 33, slice, L, W]
    return merge_seg(seg_list, True, merge)  # [batch, 33, L, W]/ [batch, 33, D, L, W]


def segcor(x:torch.Tensor, model, merge:bool=True, raw:bool=False) ->np.array:
    # x [batch, slice, 3, L, W]
    seg_list = []
    for i in range(x.shape[1]):
        seg_list.append(model(x[:, i, :]).cpu().data.numpy())  # [batch, 19, L, W]  -> [slice, batch, 19, L, W]
    seg_list = np.transpose(np.asarray(seg_list), (1, 0, 2, 3, 4))  # [slice, batch, 33, L, W] -> [batch, slice, 33, L, W]
    seg_list = post_seg(seg_list)
    if raw:
        return np.transpose(np.asarray(seg_list), (0, 2, 1, 3, 4))  # [batch, slice, 33, L, W] -> [batch, 33, slice, L, W]
    return merge_seg(seg_list, False, merge)  # [batch, 33, L, W]


def med_save(x:torch.Tensor):
    # [class, L, W]
    if len(x.shape)==3:
        c, L, W = x.shape
        Reseg = np.zeros((L, W), dtype=np.int16)
        for l in range(L):
            for w in range(W):
                Reseg[l, w] = np.argmax(x[:, l, w], axis=0)
    else:
        Reseg = x
    # Reseg [L, W]
    Reseg = Reseg * 10
    return Reseg  # Reseg [L, W]


def segmerge(seg:np.array)->np.array:
    # [batch, class, L, W]
    B, _, L, W = seg.shape
    Reseg = np.zeros((B, L, W), dtype=np.float32)
    for b in range(B):
        for l in range(1, L):
            for w in range(W):
                Reseg[b, l, w] = np.argmax(seg[b, :, l, w], axis=0)
    return Reseg  # [batch, l, w]


def overlaymerge(seg:np.array)->np.array:
    # [batch, class, L, W]
    B, C, L, W = seg.shape
    Reseg = np.zeros((B, L, W), dtype=np.float32)
    for b in range(B):
        for c in range(C-1):
            Reseg[b] += (c*seg[b, c] / C)
    return Reseg  # [batch, l, w]