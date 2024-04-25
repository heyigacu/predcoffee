# how to use 
"""
    Method 1: 
        1. make a diretory, put descriptors.txt in (run by calculate_decriptors.py)
        2. run this python file

    Method 2 :
        python check_factor_analysis.py -i /home/hy/Documents/Project/KokumiPD/Analysis/DescriptorsAnalysis/descriptors.txt
"""

import os 
import numpy as np
import pandas as pd
import argparse
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import FactorAnalyzer

DIR = os.path.abspath(os.path.dirname(__file__)) 
descriptors_filename = "all_descriptors.csv"
report_name = "check_report.txt"
descriptors_name = "descriptors_checked.csv"
parser = argparse.ArgumentParser(description='rdkit descriptors')
parser.add_argument("-i", "--inputfile", type=str, default=os.path.join(DIR,descriptors_filename),
                    help="descriptors input file, should include the headers")
parser.add_argument("-o", "--outdir", type=str, default=DIR,
                    help="out descriptors file")
parser.add_argument("-s", "--start", type=int, default=7,
                    help="the start column idx of the value matrix")
args = parser.parse_args()
print(args)


df = pd.read_csv(args.inputfile,header=0,sep='\t')
def removeTooManyZeros(df, threshold, start=0):
    """
    args:
        df <pandas DataFrame>
        threshold <int> : the number threshold of zeros, if there is a feature whose number of zeros greater than it, will remove it  
    """
    end=df.shape[-1]+1
    df_head = df.iloc[:,:start]
    df_end = df.iloc[:,end:]
    df_middle = df.iloc[:,start:end]
    zero_counts = df_middle.eq(0).sum()
    columns_to_drop = zero_counts[zero_counts > threshold].index
    # print('drop columns:',len(columns_to_drop))
    df_middle = df_middle.drop(columns_to_drop, axis=1, inplace=False)
    return pd.concat([df_head, df_middle, df_end],axis=1), df_middle.astype('float32')

def removeNanColumn(df, start=0):
    end=df.shape[-1]+1
    df_head = df.iloc[:,:start]
    df_end = df.iloc[:,end:]
    df_middle = df.iloc[:,start:end]
    df_middle = df_middle.dropna(axis=1)
    return pd.concat([df_head, df_middle, df_end],axis=1), df_middle

def removeNanRow(df, start=0):
    end=df.shape[-1]+1
    df_head = df.iloc[:,:start]
    df_end = df.iloc[:,end:]
    df_middle = df.iloc[:,start:end]
    rows_with_nan = df_middle[df_middle.isnull().any(axis=1)].index
    df = df.drop(rows_with_nan,axis=0)

    df_middle = df_middle.dropna(axis=0)
    return df, df_middle

def removeVariance(df, start=0):
    end=df.shape[-1]+1
    df_head = df.iloc[:,:start]
    df_end = df.iloc[:,end:]
    df_middle = df.iloc[:,start:end]
    # print(sorted(list(df_middle.var())))
    columns_variance_0 = df_middle.columns[df_middle.var() == 0 ]
    df_middle = df_middle.drop(columns_variance_0, axis=1)
    return pd.concat([df_head, df_middle, df_end],axis=1), df_middle


def removeGreaterThanAbsoulute(df, threshold, start=0):
    end=df.shape[-1]+1
    df_head = df.iloc[:,:start]
    df_end = df.iloc[:,end:]
    df_middle = df.iloc[:,start:end]
    columns_delete = df_middle.columns[df_middle.max() > threshold]
    # print(columns_delete)
    # print(sorted(list(df_middle.max())))
    df_middle = df_middle.drop(columns_delete, axis=1)
    return pd.concat([df_head, df_middle, df_end],axis=1), df_middle


df,df_ft= removeNanRow(df,args.start)
print(df_ft.shape)
df,df_ft= removeVariance(df,args.start)
print(df_ft.shape)
df,df_ft= removeGreaterThanAbsoulute(df, 20000, args.start)
print(df_ft.shape)
print(df.shape)
df.to_csv(os.path.join(DIR,descriptors_name),index=False,header=True,sep='\t')

def can_kmo_bartlett_test(df, start, path):
    with open(path,'w') as f:
        f.write('threshold\tfeatures\tchi_square_value\tp_value\tkmo_model\n')
        for i,threshold in enumerate(range(len(df), 0, -1)):
            try:
                print(threshold)
                df, df_ft = removeTooManyZeros(df, threshold, start)
                chi_square_value, p_value = calculate_bartlett_sphericity(df_ft)
                kmo_all, kmo_model=calculate_kmo(df_ft)
                fa = FactorAnalyzer(2,rotation=None)
                fa.fit(df_ft)
                if not np.isnan(chi_square_value) and not np.isnan(p_value) and not np.isnan(kmo_model):
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format(threshold, df_ft.shape[-1],chi_square_value, p_value, kmo_model ))
            except:
                continue
can_kmo_bartlett_test(df, args.start, os.path.join(DIR,report_name))

