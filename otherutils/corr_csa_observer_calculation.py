import os 
import pandas as pd
import numpy as np
from typing import Union, Literal
from scipy import stats


from sklearn.metrics import confusion_matrix
import os
import numpy as np
from matplotlib import pyplot as plt
import pingouin as pg
import pandas as pd


def icc_calculator(G:np.ndarray, P:np.ndarray):
    if len(G.shape)==3:
        G = np.sum(G, axis=(1,2))
        P = np.sum(P, axis=(1,2))
    elif len(G.shape)==4:
        G = np.sum(G, axis=(1,2,3))
        P = np.sum(P, axis=(1,2,3))
    elif len(G.shape)==2:
        G = np.sum(G, axis=(1))
        P = np.sum(P, axis=(1))
    index = list(range(len(G)))
    rater = [1] * len(G)
    rater2 = [0] * len(P)
    assert len(rater)==len(rater2)
    G_dict = {'ID':index, 'Score':G, 'rater':rater}
    P_dict = {'ID':index, 'Score':P, 'rater':rater2}
    Gdf = pd.DataFrame(G_dict)
    Pdf = pd.DataFrame(P_dict)

    data = pd.concat([Gdf, Pdf], axis=0)

    icc = pg.intraclass_corr(data=data, targets='ID', raters='rater', ratings='Score').round(8)# 
    icc = icc.set_index("Type")
    icc = icc.loc['ICC2']['ICC']
    return icc


def _corr_sort_distri(Garray:np.array, Parray:np.array):
    return stats.pearsonr(Garray, Parray)    


def corr_calculator(Garray:np.array, Parray:np.array, num_scores_per_site:int=43,
                    division:Union[None, int]=None, div_target:Union[int, str]=0):
    # G -- the reference, P -- the MSE or Pred.
    if len(Garray.shape) >= 2:
        if Garray.shape[1] == 1:  # only when the input array is flatten
            Garray = Garray.reshape((-1, num_scores_per_site))  # from [batch * num_scores_per_site] to [batch, num_scores_per_site]
    else:
        Garray = Garray.reshape((-1, num_scores_per_site))  # from [batch * num_scores_per_site] to [batch, num_scores_per_site]
    if len(Parray.shape) >= 2:
        if Parray.shape[1] == 1:
            Parray = Parray.reshape((-1, num_scores_per_site))
    else:
        Parray = Parray.reshape((-1, num_scores_per_site))  # from [batch * num_scores_per_site] to [batch, num_scores_per_site]
    if division:
        if type(div_target)==int:
            corr, p_value = _corr_sort_distri(Garray[:, div_target], Parray[:, div_target])
        elif type(div_target)==str:
            corr, p_value = 0, 0
            for i in range(num_scores_per_site):
                corr_i, p_value_i = _corr_sort_distri(Garray[:, i], Parray[:, i])
                corr += corr_i
                p_value += p_value_i
            corr /= num_scores_per_site
            p_value /= num_scores_per_site
        else:
            raise ValueError('not supported div target')
    else:
        corr, p_value = _corr_sort_distri(np.sum(Garray, axis=1), np.sum(Parray, axis=1))
    return corr, p_value


def corr_csa_observer(feature:Literal['TSY','SYN','BME'], score_sum:bool=False):
    csvpath = f'./output/all/1foldmerged_Wrist_{feature}_ALL_sum{score_sum}.csv'
    campath = f'./output/all/cam_use.xlsx'

    df = pd.read_csv(csvpath, index_col=0)
    cam_ids = pd.read_excel(campath)['ID'].astype(str)
    df['ID'] = df['ID'].astype(str)
    df['cam_use'] = df['ID'].isin(cam_ids).astype(int)
    df_csa = df[df['cam_use']==1]

    # column_pred, column_gt = f'Wrist_{feature}_pred', f'Wrist_{feature}_gt'
    column_pred, column_gt = f'sums', f'sums_gt'
    p_csa = df_csa[column_pred].values
    g_csa = df_csa[column_gt].values
    corr, p_value = stats.pearsonr(p_csa, g_csa)
    icc = icc_calculator(p_csa, g_csa)
    print(f'{feature} corr in csa observer study: {corr}')  
    print(f'{feature} icc in csa observer study: {icc}')   


if __name__=='__main__':
    for feature in ['TSY','SYN','BME']:
        corr_csa_observer(feature, True)


