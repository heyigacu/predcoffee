import urllib
import pubchempy
import pandas as pd
import numpy as np

smiles_path = "/home/hy/Documents/Project/odor/coffee/analysis/scratch/smiles.txt"
names_path = '/home/hy/Documents/Project/odor/coffee/analysis/scratch/names.txt'


with open(smiles_path,'r',encoding='utf-8-sig') as file1:
    file_lines = file1.readlines()
    name_list=[]
    smi_list=[]
    for i in file_lines:
        j=i.strip()
        smi_list.append(str(j))
    print(smi_list)
    for smi in  smi_list:
        print(smi)
        results = pubchempy.get_compounds(smi, 'smiles')
        for l in results:
            try:
                name_list.append(l.iupac_name)
            except (pubchempy.BadRequestError,TimeoutError,urllib.error.URLError,ValueError):
                name_list.append('error')
            dataframe=pd.DataFrame({'smi':name_list})
            dataframe.to_csv(names_path,index=False,sep='\t')
