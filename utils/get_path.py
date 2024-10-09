def getpath(path:str):
    # path: f'{self.id[idx]}_{self.date[idx]}'
    pid, ptp = path.split('_')
    return pid, ptp