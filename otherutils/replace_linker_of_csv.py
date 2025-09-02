import pandas as pd


def replace(string:str):
    id, date = string.split(';')

    return id + '&' + date

path = r'D:\ESMIRAcode\ADMIRAinfer\datasets\intermediate\csv\all_mri_init_Online.csv'

df = pd.read_csv(path)

df['ID_DATE'] = df['ID_DATE'].apply(replace)

df.to_csv(path.replace('.csv', '1.csv'))