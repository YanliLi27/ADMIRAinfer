import numpy as np
from typing import Literal
from scipy.ndimage import binary_fill_holes


# ---------------------------------------------------------- column heads ---------------------------------------------------------- #
def return_head(site, bio, return_all:bool=False):
    default_site:list={'Wrist':0, 'MCP':1, 'Foot':2}
    default_bio:list={'ERO':0, 'BME':1, 'SYN':2, 'TSY':3}
    output_keys = [
        [['WRERO1', 'WRERO2', 'WRERO3', 'WRERO4', 'WRERO5', 'WREROTM', 'WREROTD', 'WREROCA', 'WREROHA', 'WREROSC', 
          'WREROLU', 'WREROTQ', 'WREROPI', 'WRERORA', 'WREROUL'], 
         ['WRBME1', 'WRBME2', 'WRBME3', 'WRBME4', 'WRBME5', 'WRBMETM', 'WRBMETD', 'WRBMECA', 
          'WRBMEHA', 'WRBMESC', 'WRBMELU', 'WRBMETQ', 'WRBMEPI', 'WRBMERA', 'WRBMEUL'], 
         ['WRSYNRU', 'WRSYNRC', 'WRSYNIC'], 
         ['WRTSYVI', 'WRTSYV', 'WRTSYIV', 'WRTSYIII', 'WRTSYII', 'WRTSYI', 'WRTSY1', 'WRTSY2', 'WRTSY3', 'WRTSY4']],  # wrist
        [['MCDERO2', 'MCDERO3', 'MCDERO4', 'MCDERO5', 'MCPERO2', 'MCPERO3', 'MCPERO4', 'MCPERO5'], 
         ['MCDBME2', 'MCDBME3', 'MCDBME4', 'MCDBME5', 'MCPBME2', 'MCPBME3', 'MCPBME4', 'MCPBME5'], 
         ['MCSYN2', 'MCSYN3', 'MCSYN4', 'MCSYN5'], 
         ['MCFTSY2', 'MCFTSY3', 'MCFTSY4', 'MCFTSY5', 'MCETSY2', 'MCETSY3', 'MCETSY4', 'MCETSY5']],  # mcp
        [['MTDERO1', 'MTDERO2', 'MTDERO3', 'MTDERO4', 'MTDERO5', 'MTPERO1', 'MTPERO2', 'MTPERO3', 'MTPERO4', 'MTPERO5'], # MTERO
         ['MTDBME1', 'MTDBME2', 'MTDBME3', 'MTDBME4', 'MTDBME5', 'MTPBME1', 'MTPBME2', 'MTPBME3', 'MTPBME4', 'MTPBME5'], # MTBME
         ['MTSYN1', 'MTSYN2', 'MTSYN3', 'MTSYN4', 'MTSYN5'], # MTSYN
         ['MTETSY1', 'MTETSY2', 'MTETSY3', 'MTETSY4', 'MTETSY5', 'MTFTSY1', 'MTFTSY2', 'MTFTSY3', 'MTFTSY4', 'MTFTSY5']]   # MTTSY 
    ]
    if return_all:
        keys = []
        for i in range(0, 3):
            for j in range(0, 4):
                keys.extend(output_keys[i][j])  # 15+3+10+8+4+8+10+5+10
    keys = output_keys[default_site[site]][default_bio[bio]]
    return keys # list


# -------------------------------------------------------- central selector -------------------------------------------------------- #


def _normalization(data: np.array) ->np.array:  
    # return the normalized data, the mean or max will not be reserved
    max_value = np.max(data)
    min_value = np.min(data)
    data = (data - min_value) / (max_value - min_value)
    return data  # here, data will be in the range of (0, 1)

def _central_n_slices(value_array: list, num:int=5)->int:
    # value_array: [value_of_mask-1, ...] [depth, 1]
    value_array = np.array(value_array)
    five_value_array = []
    for i in range(0, len(value_array)+1-num):
        value_around = np.sum(value_array[i:i+num])
        five_value_array.append(value_around)
    max_index = five_value_array.index(max(five_value_array))
    sorted_index = sorted(range(len(five_value_array)), key=lambda k:five_value_array[k], reverse=True)
    # return a list with length = 20 - num, [a, b, c, d, e, ...] a represents the index of the largest value in array
    # the first one is the largest one
    order_array = [sorted_index.index(i) for i in range(len(sorted_index))]
    # become the order for each slice, in the order of 0-(20-num)
    return max_index, order_array


def _square_selector(data:np.array)->str:  # data: [20, 512, 512]
    assert len(data.shape) == 4 or len(data.shape) == 3 # "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[:], dtype=bool)  # [512, 512] 因为20层应该是要全用，所以说完全可以把z轴作为channel
    value_in_mask = []   # 20 slices with 20 sums
    for c in range(data.shape[0]):  # 读取单个slice
        nonzero_mask[c] = data[c] >= 0.05   # threshold = 0.1,  out of range(0, 1) after normalization
        nonzero_mask[c] = binary_fill_holes(nonzero_mask[c])  # 输入数据已经是处理后的数据了，此时无需再使用图形学
        value_in_mask.append(np.sum(nonzero_mask[c]))
    # max_range, oa = _central_n_slices(value_in_mask, num=5)
    max_range2, oa2 = _central_n_slices(value_in_mask, num=7)
    return max_range2, oa2


def _hist_selector(data:np.array)->str:  # data: [20, 512, 512]
    assert len(data.shape) == 4 or len(data.shape) == 3 # "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    # nonzero_mask = np.zeros(data.shape[:], dtype=bool)  # [512, 512] 因为20层应该是要全用，所以说完全可以把z轴作为channel
    value_in_mask = []   # 20 slices with 20 sums
    for c in range(data.shape[0]):  # 读取单个slice
        hist, _ = np.histogram(data[c], bins=20, range=(0,1))
        value_in_mask.append(-np.std(hist))  # 取std作为均衡化的标准，越小越好因此取负值
    # max_range, oa = _central_n_slices(value_in_mask, num=5)
    max_range2, oa2 = _central_n_slices(value_in_mask, num=7)
    return max_range2, oa2


def central_selector(data_array:np.array)->str:
    # datapath: os.path.join(dirpath, file) -- data_root + subnaem + filename = 'Root/EAC_Wrist_TRA/Names'
    if data_array.shape[0] < 7:
        raise ValueError(f'image have shape of {data_array.shape}')
    data_array = _normalization(data_array)  # 可以使用class内的normalization
    square_mr2, square_oa2 = _square_selector(data_array)  # [slice, 512, 512] the mask for each slice
    hist_mr2, hist_oa2 = _hist_selector(data_array)
    if square_mr2==hist_mr2:
        max_range2 = square_mr2
    else:
        oa2 = list(np.asarray(square_oa2) + np.asarray(hist_oa2))
        max_range2 = oa2.index(min(oa2))

    return max_range2
