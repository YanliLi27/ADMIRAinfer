
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
            for j in range(1, 3):
                keys.extend(output_keys[i][j])  # 15+3+10+8+4+8+10+5+10
    keys = output_keys[default_site[site]][default_bio[bio]]
    return keys # list



def return_head_gt(site, bio, return_all:bool=False):
    default_site:list={'Wrist':0, 'MCP':1, 'Foot':2}
    default_bio:list={'ERO':0, 'BME':1, 'SYN':2, 'TSY':3}
    output_keys = [
        [['WRERO1_GT', 'WRERO2_GT', 'WRERO3_GT', 'WRERO4_GT', 'WRERO5_GT', 'WREROTM_GT', 'WREROTD_GT', 'WREROCA_GT', 'WREROHA_GT', 'WREROSC_GT', 
          'WREROLU_GT', 'WREROTQ_GT', 'WREROPI_GT', 'WRERORA_GT', 'WREROUL_GT'], 
         ['WRBME1_GT', 'WRBME2_GT', 'WRBME3_GT', 'WRBME4_GT', 'WRBME5_GT', 'WRBMETM_GT', 'WRBMETD_GT', 'WRBMECA_GT', 
          'WRBMEHA_GT', 'WRBMESC_GT', 'WRBMELU_GT', 'WRBMETQ_GT', 'WRBMEPI_GT', 'WRBMERA_GT', 'WRBMEUL_GT'], 
         ['WRSYNRU_GT', 'WRSYNRC_GT', 'WRSYNIC_GT'], 
         ['WRTSYVI_GT', 'WRTSYV_GT', 'WRTSYIV_GT', 'WRTSYIII_GT', 'WRTSYII_GT', 'WRTSYI_GT', 'WRTSY1_GT', 'WRTSY2_GT', 'WRTSY3_GT', 'WRTSY4_GT']],  # wrist
        [['MCDERO2_GT', 'MCDERO3_GT', 'MCDERO4_GT', 'MCDERO5_GT', 'MCPERO2_GT', 'MCPERO3_GT', 'MCPERO4_GT', 'MCPERO5_GT'], 
         ['MCDBME2_GT', 'MCDBME3_GT', 'MCDBME4_GT', 'MCDBME5_GT', 'MCPBME2_GT', 'MCPBME3_GT', 'MCPBME4_GT', 'MCPBME5_GT'], 
         ['MCSYN2_GT', 'MCSYN3_GT', 'MCSYN4_GT', 'MCSYN5_GT'], 
         ['MCFTSY2_GT', 'MCFTSY3_GT', 'MCFTSY4_GT', 'MCFTSY5_GT', 'MCETSY2_GT', 'MCETSY3_GT', 'MCETSY4_GT', 'MCETSY5_GT']],  # mcp
        [['MTDERO1_GT', 'MTDERO2_GT', 'MTDERO3_GT', 'MTDERO4_GT', 'MTDERO5_GT', 'MTPERO1_GT', 'MTPERO2_GT', 'MTPERO3_GT', 'MTPERO4_GT', 'MTPERO5_GT'], # MTERO
         ['MTDBME1_GT', 'MTDBME2_GT', 'MTDBME3_GT', 'MTDBME4_GT', 'MTDBME5_GT', 'MTPBME1_GT', 'MTPBME2_GT', 'MTPBME3_GT', 'MTPBME4_GT', 'MTPBME5_GT'], # MTBME
         ['MTSYN1_GT', 'MTSYN2_GT', 'MTSYN3_GT', 'MTSYN4_GT', 'MTSYN5_GT'], # MTSYN
         ['MTETSY1_GT', 'MTETSY2_GT', 'MTETSY3_GT', 'MTETSY4_GT', 'MTETSY5_GT', 'MTFTSY1_GT', 'MTFTSY2_GT', 'MTFTSY3_GT', 'MTFTSY4_GT', 'MTFTSY5_GT']]   # MTTSY 
    ]
    
    if return_all:
        keys = []
        for i in range(0, 3):
            for j in range(1, 3):
                keys.extend(output_keys[i][j])  # 15+3+10+8+4+8+10+5+10
    keys = output_keys[default_site[site]][default_bio[bio]]
    return keys # list