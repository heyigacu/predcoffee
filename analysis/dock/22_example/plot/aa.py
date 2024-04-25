
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = os.path.abspath(os.path.dirname(__file__)) 
plif_path = base_dir+'/../resin_7A.txt'
aa_path = base_dir+'/../aa.txt'
ls = []
with open(plif_path,'r') as f:
    lines = f.readlines()
    for line in lines:
        ls.append(line.strip())


ls = sorted(set(ls), key=ls.index)
with open(aa_path,'w') as f:
    for i in ls:
        f.write(i+'\n')