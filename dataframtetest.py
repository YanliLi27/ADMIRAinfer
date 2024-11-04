from utils.get_head import return_head, return_head_gt
import pandas as pd
from typing import Literal, Optional, List

def main_process(task:Literal['CSA', 'TE'], site:Literal['Wrist','MCP','Foot'],
                 feature:Literal['TSY','SYN','BME'], 
                 view:Optional[List[str]]=['TRA', 'COR'],
                 order:int=0, score_sum:bool=False, filt:Optional[list]=None):
    res_head= ['ID', 'ScanDatum', 'ID_Timepoint']
    res_head.extend(return_head(site, feature))
    res_head.extend(return_head_gt(site, feature))
    df = pd.DataFrame(index=range(100), columns=res_head)
    print(1)


if __name__=='__main__':
    for site in ['Wrist', 'MCP', 'Foot']:
        for feature in ['TSY','SYN','BME']:
            main_process('CSA', site, feature, view=['TRA', 'COR'], order=0, score_sum=False, filt=None)