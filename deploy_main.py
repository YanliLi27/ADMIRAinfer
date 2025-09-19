from deployment.quant_main.admira_agent import ADMIRAquant
import SimpleITK as sitk
import numpy as np
import argparse
from typing import Literal, List
import sys


if __name__=='__main__':
    tra_path:str = r'E:\ESMIRA_RAprediction\Export20Jun22\EAC_Wrist_TRA\ESMIRA-LUMC-Arth2848_EAC-20100824-RightWrist_PostTRAT1f_1.mha'
    cor_path:str = r'E:\ESMIRA_RAprediction\Export20Jun22\EAC_Wrist_COR\ESMIRA-LUMC-Arth2848_EAC-20100824-RightWrist_PostCORT1f_1.mha'
    tra_data = sitk.ReadImage(tra_path)
    tra_data:np.ndarray = sitk.GetArrayFromImage(tra_data)
    cor_data = sitk.ReadImage(cor_path)
    cor_data:np.ndarray = sitk.GetArrayFromImage(cor_data)
    data = {'TRAT1f': tra_data, 'CORT1f': cor_data}

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_anatomical_site",
        choices=['Wrist', 'MCP', 'Foot'],
        help="The anatomical site that wanted to be quantified",
        default='Wrist',
    )
    parser.add_argument(
        "--target_inflammation_feature",
        choices=['TSY', 'SYN', 'BME'],
        help="The inflammation feature that wanted to be quantified",
        default='TSY',
    )
    parser.add_argument(
        "--quantification_type",
        choices=['Total', 'PerLocation'],
        help="The output type - either one score for the entire site and feature, or one score for each location",
        default='PerLocation',
    )
    parser.add_argument(
        "--model_type",
        choices=[['TRA'], ['COR'], ['TRA', 'COR']],
        help="The model take-in type - either TRA or COR or TRA+COR",
        default=['TRA', 'COR'],
    )

    args = parser.parse_args()



    results = ADMIRAquant(data, args)

    print(results)