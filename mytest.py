from datasets.get_data import getdata

if __name__=='__main__':
    getdata('CSA', 'Wrist', 'TSY', ['TRA', 'COR'])
    getdata('TE', 'Wrist', 'TSY', ['TRA', 'COR'])
