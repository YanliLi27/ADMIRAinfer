import sys
import os
import argparse
from pathlib import Path
from typing import Union, Optional, Literal, Any, List  # Any for namespace

import SimpleITK as sitk
import numpy as np

from .admira_agent import ADMIRAquant


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


def main():
    # -------------------------------------------------------------------------------
    # TODO use --target_anatomical_site='Wrist', --target_inflammation_feature=Literal['TSY', 'SYN', 'BME'], 
    # --quantification_type='Total', --model_type=['TRA'] to test one view model
    # -------------------------------------------------------------------------------

    input_directory = Path(os.environ['INPUT_FOLDER']) # input directory for DICOM
    input_model = Path(os.environ['INPUT_MODEL']) # input directory for AI model
    # model stored at: input_model / 'Total' or 'PerLocation' / f'{feature}__{site}_{mt}_fold{fold}Sum.onnx' or f'{feature}__{site}_{mt}_fold{fold}.onnx'
    # feature - 'TSY', 'SYN', 'BME', site - 'Wrist', 'MCP', 'Foot', mt - '2dirc', 'TRA', 'COR', fold - 0->5
    output_directory = Path(os.environ['OUTPUT_FOLDER']) # output directory for AI, folder where Pandas file is stored


    # -------------------------------------- argparse -----------------------------------------
    parser = argparse.ArgumentParser(
        description="Read a folder of DICOM images with SimpleITK."
    )
    parser.add_argument(
        "target_anatomical_site",
        type=Literal['Wrist', 'MCP', 'Foot'],
        help="The anatomical site that wanted to be quantified",
        default='Wrist',
    )
    parser.add_argument(
        "target_inflammation_feature",
        type=Literal['TSY', 'SYN', 'BME'],
        help="The inflammation feature that wanted to be quantified",
        default='TSY',
    )
    parser.add_argument(
        "quantification_type",
        type=Literal['Total', 'PerLocation'],
        help="The output type - either one score for the entire site and feature, or one score for each location",
        default='Total',
    )
    parser.add_argument(
        "model_type",
        type=List[str], # ['TRA'], ['COR'], ['TRA', 'COR']
        help="The model take-in type - either TRA or COR or TRA+COR",
        default=['TRA'],
    )
    args = parser.parse_args()
    # -------------------------------------- argparse -----------------------------------------

    # ------------------------------------ data loader ----------------------------------------
    try:
        itk_image:dict[str, np.ndarray] = read_dicom_series(input_directory)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------------------
    # TODO itk_image{'TRAT1f': data-np.ndarray, Optional('CORT1f': data-np.ndarray)}
    # -------------------------------------------------------------------------------
    # ------------------------------------ data loader ----------------------------------------

    result:dict[str, float] = ADMIRAquant(itk_image, input_model, args)
    # result: {'Total inflammation': float} or {'Site_Feature': float, ...}

    # save_result(result, output_directory)
    print(result)


if __name__ == "__main__":
    main()

