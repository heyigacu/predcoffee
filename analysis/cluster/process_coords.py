import numpy as np
import pandas as  pd

arr = np.loadtxt("/home/hy/Documents/Project/odor/coffee/analysis/cluster/tsne_coordinate.txt")
df = pd.DataFrame(arr, columns=['tSNE1','tSNE2'])

df.round(pd.Series([2,2], index=['tSNE1', 'tSNE2'])).to_csv("/home/hy/Documents/Project/odor/coffee/analysis/cluster/tsne_coordinate_.txt",index=False, sep="\t",header=True)