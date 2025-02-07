# 把列表转化为需要计算的结果，并返回img, gt, path
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from torch.utils import data
from skimage.transform import resize
import torch
from typing import Union


class CLIPDataset3D(data.Dataset):
    def __init__(self, df:pd.DataFrame, path_column:list, score_column:list,
                 score_sum:bool=False, path_flag:bool=False):
        df = df.reset_index(drop=True)
        self.df:pd.DataFrame = df
        self.id = df['ID'].values
        self.date = df['DATE'].values
        self.path_column:list = path_column
        self.score_column:list = score_column

        self.transform = None
        self.score_sum = score_sum
        self.slices = 7
        self.full_img = False
        self.path_flag = path_flag

    
    def __len__(self):
        return self.id.shape[0]

    def __getitem__(self, idx):
        path = f'{self.id[idx]}_{int(self.date[idx])}'
        if self.full_img:
            data = self._load_full(idx)  # data list [scan-tra, scan-cor]
        else:
            data = self._load_file(idx)  # data list [scan-tra, scan-cor]
        # data list[N, array[5/7/20, 512, 512]], path
        scores = np.sum(np.asarray([self.df.loc[idx, self.score_column].to_numpy()], dtype=np.float32), axis=1) if self.score_sum \
            else self.df.loc[idx, self.score_column].to_numpy().astype(float)
        
        for i in range(len(data)):
            data[i] = torch.from_numpy(data[i])
            if self.transform is not None:
                data[i] = self.transform(data[i])
        # data list [tensors]
        data = torch.stack(data)  # [Site*TRA/COR, slice/depth, length, width]

        if self.path_flag:
            return data, scores, path
        return data, scores 


    def _load_file(self, idx):
        data_matrix = []
        paths = self.df.loc[idx, self.path_column].to_numpy()
        for indiv_path in paths:
            # indiv_path: 'subdir\names.mha:cs'
            # updated: 'subdir\names.mha:1to6plus1to11'
            try:
                path, cs =  indiv_path.split('[')
            except:
                try:
                    first, sec, cs = indiv_path.split('[')
                    path = first + '[' + sec
                except:
                    raise ValueError(f'wrong path: {indiv_path} from idx: {paths}')
            lower, upper = cs.split('to')
            lower, upper = int(lower), int(upper)

            data_mha = sitk.ReadImage(path)
            data_array = sitk.GetArrayFromImage(data_mha)

            # for COR，using lower and upper
            if 'CORT1f' in path:
                data_array = self._itensity_normalize(data_array[lower:upper])
            # for TRA, using step
            elif 'TRAT1f' in path:
                if data_array.shape[0]//2 >= self.slices:
                    center = data_array.shape[0]//2
                    s = slice(center-self.slices, center+self.slices, 2)
                    data_array = self._itensity_normalize(data_array[s])
                else:
                    data_array = self._itensity_normalize(data_array[-7:])
            # [5, 512, 512]/[10, 512, 512]
            if data_array.shape != (self.slices, 512, 512):
                if data_array.shape == (self.slices, 256, 256):
                    data_array = resize(data_array, (self.slices, 512, 512), preserve_range=True)  # preserve_range: no normalization
                else:
                    raise ValueError('the shape of input:{}, the id: {}, central_slice: {}'.format(data_array.shape, path, lower))
            data_matrix.append(data_array.astype(np.float32))
        return data_matrix  # [N, 5, 512, 512]


    def _load_full(self, idx):
        data_matrix = []
        paths = self.df.loc[idx, self.path_column].to_numpy()
        for indiv_path in paths:
            # indiv_path: 'subdir\names.mha:cs'
            path, _ = indiv_path.split(']')  # 'subdir\names.mha', 'cs'
            data_mha = sitk.ReadImage(path)
            data_array = sitk.GetArrayFromImage(data_mha)
            data_array = self._itensity_normalize(data_array)  # [20, 512, 512]
            if data_array.shape != (20, 512, 512):
                data_array = resize(data_array, (20, 512, 512), preserve_range=True)  # preserve_range: no normalization      
            data_matrix.append(data_array.astype(np.float32))
        return data_matrix  # [N, 20, 512, 512]


    def _itensity_normalize(self, volume: np.array):
        """
        normalize the itensity of a volume based on the mean and std of nonzeor region
        inputs:
            volume: the input volume
        outputs:
            out: the normalized volume
        """
        min_value = volume.min()
        max_value = volume.max()
        if max_value > min_value:
            out = (volume - min_value) / (max_value - min_value)
        else:
            out = volume
        # out_random = np.random.normal(0, 1, size=volume.shape)
        # out[volume == 0] = out_random[volume == 0]
        return out