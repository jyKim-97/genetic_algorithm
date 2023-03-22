import matplotlib.pyplot as plt
import pickle as pkl
import os
import numpy as np 


class Logger:
    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self._read_log()
        self.load_param_id = -1
        
    def _read_log(self):
        self.fit_scores, self.job_ids = read_log(os.path.join(self.parent_dir, "log.txt"))
    
    def view_log(self, nstart=0, f=np.average):
        s = f(self.fit_scores, axis=1)
        
        plt.figure(dpi=120, figsize=(4,4))
        plt.plot(s[nstart:], 'k.-')
        plt.xlabel("epoch", fontsize=20)
        plt.ylabel("fitness", fontsize=20)
        plt.show()

        print(len(s))
        
    def load_params(self, param_id):
        nlog = len(self.fit_scores)
        if param_id >= nlog:
            print("param_id exceeds nlogs: %d"%(nlog))
        
        with open(os.path.join(self.parent_dir, "params_%d.pkl"%(param_id)), "rb") as fid:
            data = pkl.load(fid)
            self.job_id_set = data["job_id"]
            self.param_set = data["params"]
            
        self.load_param_id = param_id
            

def read_log(log_fname):
    fit_scores = []
    job_ids = []
    with open(log_fname, "r") as fid:
        line = fid.readline()
        while line:
            fit_scores.append([])
            job_ids.append([])
            for tmp in line.split(",")[:-1]:
                p = tmp.split(":")
                fit_scores[-1].append(float(p[1]))
                job_ids[-1].append(int(p[0]))
            line = fid.readline()
    return fit_scores, job_ids