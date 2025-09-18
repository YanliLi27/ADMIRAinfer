from deployment.quant_main.create_onnx import create_onnx_from_model
from pathlib import Path
import os


if __name__=='__main__':
    # for site in ['Wrist', 'MCP', 'Foot']:
    #     for fea in ['TSY', 'SYN', 'BME']:
    #         for model_type in [['TRA'], ['TRA', 'COR']]:
    #             for score_sum in [True, False]:
    #                 for fold in [0]:
    #                     model_dir:str = r'R:\ESMIRA\ESMIRA_Models\ADMIRA\onnx_model\20250918'
    #                     mt = '2dirc' if len(model_type)>1 else f'{model_type[0]}'
    #                     model_path:str = Path(model_dir / 'onnx_model' / '20250918' / 'Total' / f'{fea}__{site}_{mt}_fold{fold}Sum.onnx')
    #                     src:str = model_path.replace('onnx_model', 'raw_model').replace('.model', '.onnx')
    #                     if Path.exists(src):
    #                         create_onnx_from_model(src, model_path, site, fea, model_type, score_sum)
    #                     else:
    #                         print(f'{src} does not exists')

    # for PACS system tests - one-view model
    for site in ['Wrist']:
        for fea in ['TSY', 'SYN', 'BME']:
            for model_type in [['TRA'], ['COR']]:
                for score_sum in [True]:
                    for fold in [0]:
                        model_dir:str = r'R:\ESMIRA\ESMIRA_Models\ADMIRA\onnx_model\20250918'
                        mt = '2dirc' if len(model_type)>1 else f'{model_type[0]}'
                        model_path:str = Path(model_dir)
                        model_path:str = str(model_path / 'Total' / f'{fea}__{site}_{mt}_fold{fold}Sum.onnx')
                        src:str = model_path.replace('onnx_model', 'raw_model')
                        src:str = src.replace('.onnx', '.model')
                        if os.path.exists(src):
                            create_onnx_from_model(src, model_path, site, fea, model_type, score_sum)
                        else:
                            print(f'{src} does not exists')