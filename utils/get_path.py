def getpath(path:str):
    # path: f'{self.id[idx]};{self.date[idx]}'
    pid, ptp = path.split(';')
    return pid, ptp