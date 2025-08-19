# copy from:
# R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Csa839_CSA\20210428\ESMIRA-LUMC-Csa839_CSA-20210428-RAMRIS.xml
# to:
# R:\ESMIRA\ESMIRA_Results\CAM_for_RAMRIS_estimation_Yanli_2view_07112024\IDCsa001_20120411_viewTRA_origin.nii.gz -> 
# -> R:\ESMIRA\ESMIRA_Results\CAM_for_RAMRIS_estimation_Yanli_2view_07112024\xmlfiles\ID_Date_xml

import shutil
import os

# 从R:\ESMIRA\ESMIRA_Results\CAM_for_RAMRIS_estimation_Yanli_2view_07112024\下读取所有可能的ID + Date组合，然后根据这个去
# R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_ID_CSA\Date\ 下找xml文件


def copy_paste(dir:str=r'R:\ESMIRA\ESMIRA_Results\CAM_for_RAMRIS_estimation_Yanli_2view_07112024\ramris_siteWrist_feaTSY_TRA',
               xmldir:str=r'R:\ESMIRA\ESMIRA_Database\LUMC',
               newxmldir:str=r'R:\ESMIRA\ESMIRA_Results\CAM_for_RAMRIS_estimation_Yanli_2view_07112024\xmlfiles'):
    # dir: R:\ESMIRA\ESMIRA_Results\CAM_for_RAMRIS_estimation_Yanli_2view_07112024
    if not os.path.exists(newxmldir):
        os.makedirs(newxmldir)
    allfiles:list = os.listdir(dir)
    originfiles:list = [file for file in allfiles if '_origin.nii.gz' in file]  #  IDCsa001_20120411_viewTRA_origin.nii.gz
    # id_date:list = []
    for file in originfiles:
        spliter = file.split('_') #  IDCsa001_20120411_viewTRA_origin.nii.gz -> IDCsa001, 20120411, ...
        id = spliter[0][2:]
        date = spliter[1]
        # id_date.append([id, date])

    # 根据这些id,date去找数据库里面的xml
        xmlpath = os.path.join(xmldir, f'ESMIRA_patient_{id}_CSA', f'{date}')
        xmlfiles = os.listdir(xmlpath)
        xmlfilelist = [file for file in xmlfiles if '-RAMRIS.xml' in file]
        xmlfile = xmlfilelist[0]  #  ESMIRA-LUMC-Csa839_CSA-20210428-RAMRIS.xml
        absxmlfile = os.path.join(xmlpath, xmlfile)  
        # 'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_{id}_CSA\{date}' + 'ESMIRA-LUMC-Csa839_CSA-20210428-RAMRIS.xml'
        assert os.path.exists(absxmlfile)
    # 复制到文件下
        newxmlfile:str = os.path.join(newxmldir, f'ID{id}_{date}_RAMRIS.xml')
        # 'R:\ESMIRA\ESMIRA_Results\CAM_for_RAMRIS_estimation_Yanli_2view_07112024\xmlfiles' + f'ID{id}_{date}_RAMRIS.xml'
        shutil.copyfile(absxmlfile, newxmlfile)


def copy_paste_eac(dir:str=r'R:\ESMIRA\ESMIRA_Results\CAM_for_RAMRIS_estimation_Yanli_2view_07112024\ramris_siteWrist_feaTSY_TRA',
               xmldir:str=r'R:\ESMIRA\ESMIRA_Database\LUMC',
               newxmldir:str=r'R:\ESMIRA\ESMIRA_Results\CAM_for_RAMRIS_estimation_Yanli_2view_07112024\xmlfiles'):
    # dir: R:\ESMIRA\ESMIRA_Results\CAM_for_RAMRIS_estimation_Yanli_2view_07112024
    if not os.path.exists(newxmldir):
        os.makedirs(newxmldir)
    allfiles:list = os.listdir(dir)
    originfiles:list = [file for file in allfiles if ('_origin.nii.gz' in file and 'Arth' in file)]  #  IDCsa001_20120411_viewTRA_origin.nii.gz
    # id_date:list = []
    print(len(originfiles))
    for file in originfiles:
        spliter = file.split('_') #  IDCsa001_20120411_viewTRA_origin.nii.gz -> IDCsa001, 20120411, ...
        id = spliter[0][2:]
        date = spliter[1]
        # id_date.append([id, date])

    # 根据这些id,date去找数据库里面的xml
        # ESMIRA-LUMC-Arth4420_EAC-20171020-RAMRIS.xml
        xmlpath = os.path.join(xmldir, f'ESMIRA_patient_{id}_EAC', f'{date}')
        xmlfiles = os.listdir(xmlpath)
        xmlfilelist = [file for file in xmlfiles if '-RAMRIS.xml' in file]
        xmlfile = xmlfilelist[0]  #  ESMIRA-LUMC-Csa839_CSA-20210428-RAMRIS.xml
        absxmlfile = os.path.join(xmlpath, xmlfile)  
        # 'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_{id}_CSA\{date}' + 'ESMIRA-LUMC-Csa839_CSA-20210428-RAMRIS.xml'
        assert os.path.exists(absxmlfile)
    # 复制到文件下
        newxmlfile:str = os.path.join(newxmldir, f'ID{id}_{date}_RAMRIS.xml')
        # 'R:\ESMIRA\ESMIRA_Results\CAM_for_RAMRIS_estimation_Yanli_2view_07112024\xmlfiles' + f'ID{id}_{date}_RAMRIS.xml'
        shutil.copyfile(absxmlfile, newxmlfile)



# copy_paste()
copy_paste_eac()