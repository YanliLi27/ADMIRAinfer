import pandas as pd


def get_projection(save:bool=False):
    longitudinal:list = [
        # MT --------------------------------------------------------------
        'MTD1ERO', 'MTD2ERO', 'MTD3ERO', 'MTD4ERO', 'MTD5ERO',  # MTERO - DE
        'MTP1ERO',  'MTP2ERO', 'MTP3ERO', 'MTP4ERO', 'MTP5ERO', # MTERO - PE
        
        'MTD1BME', 'MTD2BME', 'MTD3BME', 'MTD4BME', 'MTD5BME',  # MTBME - D
        'MTP1BME', 'MTP2BME', 'MTP3BME', 'MTP4BME', 'MTP5BME',  # MTBME - P
        
        'MT1SYN', 'MT2SYN', 'MT3SYN', 'MT4SYN',  'MT5SYN',  # MTSYN

        'MT1ET', 'MT2ET', 'MT3ET', 'MT4ET', 'MT5ET',  # MTTSY - E
        'MT1FT', 'MT2FT', 'MT3FT',  'MT4FT', 'MT5FT',  # MTTSY - F
    
        # MC --------------------------------------------------------------
        'MCD2ERO', 'MCD3ERO', 'MCD4ERO', 'MCD5ERO',   # MCERO - D
        'MCP2ERO', 'MCP3ERO', 'MCP4ERO', 'MCP5ERO',   # MCERO - P

        'MCD2BME', 'MCD3BME', 'MCD4BME', 'MCD5BME',   # MCBME - D
        'MCP2BME', 'MCP3BME', 'MCP4BME', 'MCP5BME',   # MCBME - P

        'MC2SYN',  'MC3SYN', 'MC4SYN', 'MC5SYN',   # MCSYN
        
        'MC2ETS', 'MC3ETS', 'MC4ETS', 'MC5ETS', # MCTSY - E
        'MC2FTS', 'MC3FTS', 'MC4FTS', 'MC5FTS', # MCTSY - F
        
        
        # WR --------------------------------------------------------------
        'WR1ERO', 'WR2ERO', 'WR3ERO', 'WR4ERO', 'WR5ERO',        # WRERO - D
        'WRtmERO', 'WRtdERO', 'WRcaERO', 'WRhaERO', 'WRscERO',   # WRERO - TM, TD, CA, HA, SC
        'WRluERO', 'WRtrERO', 'WRpiERO', 'WRraERO', 'WRulERO',   # WRERO - LU, TQ, PI, RA, UL
        
        
        'WR1BME', 'WR2BME', 'WR3BME', 'WR4BME',  'WR5BME',       # WRBME
        'WRtmBME', 'WRtdBME', 'WRcaBME', 'WRhaBME', 'WRscBME',   # WRBME - TM, TD, CA, HA, SC
        'WRluBME', 'WRtrBME', 'WRpiBME', 'WRraBME', 'WRulBME',   # WRBME - LU, TQ, PI, RA, UL
        
        
        'WRruSYN', 'WRrcSYN', 'WRicSYN',   # WRSYN
    
        'WRETSVI', 'WRETSV', 'WRETSIV', 'WRETSIII', 'WRETSII', 'WRETSI', # WRTSY
        'WRFTS1', 'WRFTS2', 'WRFTS3', 'WRFTS4'                           # WRTSY
        ]


    previous = [
        # MT --------------------------------------------------------------
        'MTDERO1', 'MTDERO2', 'MTDERO3', 'MTDERO4', 'MTDERO5', # MTERO - DE
        'MTPERO1', 'MTPERO2', 'MTPERO3', 'MTPERO4', 'MTPERO5', # MTERO - PE

        'MTDBME1', 'MTDBME2', 'MTDBME3', 'MTDBME4', 'MTDBME5', # MTBME - D
        'MTPBME1', 'MTPBME2', 'MTPBME3', 'MTPBME4', 'MTPBME5', # MTBME - P

        'MTSYN1', 'MTSYN2', 'MTSYN3', 'MTSYN4', 'MTSYN5', # MTSYN

        'MTETSY1', 'MTETSY2', 'MTETSY3', 'MTETSY4', 'MTETSY5',   # MTTSY - E
        'MTFTSY1', 'MTFTSY2', 'MTFTSY3', 'MTFTSY4', 'MTFTSY5',    # MTTSY - F

        # MC --------------------------------------------------------------
        'MCDERO2', 'MCDERO3', 'MCDERO4', 'MCDERO5',   # MCERO - D
        'MCPERO2', 'MCPERO3', 'MCPERO4', 'MCPERO5',   # MCERO - P

        'MCDBME2', 'MCDBME3', 'MCDBME4', 'MCDBME5',   # MCBME - D
        'MCPBME2', 'MCPBME3', 'MCPBME4', 'MCPBME5',   # MCBME - P

        'MCSYN2', 'MCSYN3', 'MCSYN4', 'MCSYN5',    # MCSYN

        'MCETSY2', 'MCETSY3', 'MCETSY4', 'MCETSY5',   # MCTSY - E
        'MCFTSY2', 'MCFTSY3', 'MCFTSY4', 'MCFTSY5',   # MCTSY - F
        

        # WR --------------------------------------------------------------
        'WRERO1', 'WRERO2', 'WRERO3', 'WRERO4', 'WRERO5',       # WRERO - D
        'WREROTM', 'WREROTD', 'WREROCA', 'WREROHA', 'WREROSC',  # WRERO - TM, TD, CA, HA, SC
        'WREROLU', 'WREROTQ', 'WREROPI', 'WRERORA', 'WREROUL',  # WRERO - LU, TQ, PI, RA, UL

        'WRBME1', 'WRBME2', 'WRBME3', 'WRBME4', 'WRBME5',       # WRBME
        'WRBMETM', 'WRBMETD', 'WRBMECA', 'WRBMEHA', 'WRBMESC',  # WRBME - TM, TD, CA, HA, SC
        'WRBMELU', 'WRBMETQ', 'WRBMEPI', 'WRBMERA', 'WRBMEUL',  # WRBME - LU, TQ, PI, RA, UL
            
        'WRSYNRU', 'WRSYNRC', 'WRSYNIC',   # WRSYN

        'WRTSYVI', 'WRTSYV', 'WRTSYIV', 'WRTSYIII', 'WRTSYII', 'WRTSYI',   # WRTSY
        'WRTSY1', 'WRTSY2', 'WRTSY3', 'WRTSY4'                            # WRTSY
    ]

    # WR1 = [item for item in longitudinal if 'WR' in item[:2]]
    # WR2 = [item for item in previous if 'WR' in item[:2]]
    # MC1 = [item for item in longitudinal if 'MC' in item[:2]]
    # MC2 = [item for item in previous if 'MC' in item[:2]]
    # MT1 = [item for item in longitudinal if 'MT' in item[:2]]
    # MT2 = [item for item in previous if 'MT' in item[:2]]

    if save:
        df = pd.DataFrame({'longitudinal':longitudinal , 'previous':previous})
        df.to_csv(r"Projection_table.csv")

    projection:dict = {}
    for item in zip(*[longitudinal, previous]):
        projection[item[0]] = item[1]
    
    return projection


if __name__=="__main__":
    get_projection()