from typing import Literal


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
