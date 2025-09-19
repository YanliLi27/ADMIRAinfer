import argparse
import os

import SimpleITK as sitk
import numpy as np
from loguru import logger
from pathlib import Path

from admira_agent import ADMIRAquant

def read_dicom_series(directory: Path) -> sitk.Image:
    """
    Reads a DICOM series from the given directory using SimpleITK.
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a valid directory")

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(directory))

    if not dicom_names:
        raise FileNotFoundError(f"No DICOM series found in directory: {directory}")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def quantification(input_directory: Path, input_models:Path):
    #tra_path: str = r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth2848_EAC\20100824\RightWrist_PostTRAT1f\images\itk\ESMIRA-LUMC-Arth2848_EAC-20100824-RightWrist_PostTRAT1f.mha'
    #cor_path: str = r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth2848_EAC\20100824\RightWrist_PostCORT1f\images\itk\ESMIRA-LUMC-Arth2848_EAC-20100824-RightWrist_PostCORT1f.mha'

    tra_data = read_dicom_series(input_directory) # sitk.ReadImage(tra_path)
    tra_data = sitk.GetArrayFromImage(tra_data)

    # cor_data = sitk.ReadImage(cor_path)
    # cor_data: np.ndarray = sitk.GetArrayFromImage(cor_data)
    ##data = {'TRAT1f': tra_data, 'CORT1f': cor_data}
    data = {'TRAT1f': tra_data}

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_anatomical_site",
        choices=['Wrist'], #, 'MCP', 'Foot'],
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
        #default='PerLocation', # that is with data = {'TRAT1f': tra_data, 'CORT1f': cor_data}
        default='Total',
    )
    parser.add_argument(
        "--model_type",
        choices=[['TRA'], ['COR'], ['TRA', 'COR']],
        help="The model take-in type - either TRA or COR or TRA+COR",
        #default=['TRA', 'COR'],
        default=['TRA'], # that is with data = {'TRAT1f': tra_data, 'CORT1f': cor_data}
    )

    args = parser.parse_args()

    results = ADMIRAquant(data, args, model_dir=str(input_models))

    print(results)


def main() -> None:
    # Hardcoded in algorithm_dockers\aorta_aneurysm\main.py
    input_directory = Path(os.environ['INPUT_FOLDER'])
    input_models = Path(os.environ['INPUT_MODELS'])
    output_directory = Path(os.environ['OUTPUT_FOLDER'])

    logger.remove()  # remove the default stderr logger
    logger.add(output_directory / 'wrist_admira.log', level='TRACE', mode='w',
               format='{time:%Y-%m-%d_%H:%M:%S.%f} | {level.icon} | {message} | {name}:{file}:{function}[{line}]')

    # nifti(input_directory, output_directory / 'output.nii')
    logger.info('Creating Quantification Object.')
    result = quantification(input_directory, input_models)
    print(result)
    #logger.info('Creating Structured Report.')


if __name__ == "__main__":
    main()


