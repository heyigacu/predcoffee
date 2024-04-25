import os 
import numpy as np
import pandas as pd
import argparse
from scipy.stats import ttest_1samp
from scipy import stats
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn.preprocessing import MinMaxScaler
DIR = os.path.abspath(os.path.dirname(__file__)) 
descriptors_filename = "all_descriptors.csv"
parser = argparse.ArgumentParser(description='rdkit descriptors')
parser.add_argument("-i", "--inputfile", type=str, default=os.path.join(DIR,descriptors_filename),
                    help="descriptors input file, should include the headers")
parser.add_argument("-o", "--outdir", type=str, default=DIR,
                    help="out directory")
parser.add_argument("-t", "--threshold", type=int, default=0,
                    help="the number threshold of zeros,")
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
    print('drop columns:',len(columns_to_drop))
    df_middle = df_middle.drop(columns_to_drop, axis=1, inplace=False)
    return pd.concat([df_head, df_middle, df_end],axis=1), df_middle.astype('float32')

plt.rc('xtick', labelsize= 15)
plt.rc('ytick', labelsize= 15) 
plt.rc('font',weight='black')
plt.rc('xtick.major',width=1.5)
plt.rc('ytick.major',width=1.5)
plt.rc('axes',labelweight='bold',labelsize=20, titlesize=17, titleweight='bold',linewidth=1.5)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

def scree_plt_raw(ev):
    plt.figure(figsize=(8,6))
    plt.scatter(range(1, len(ev) + 1), ev, s=15)
    plt.plot(range(1, len(ev) + 1), ev, markersize= 10)
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalue")
    plt.grid()
    plt.savefig(os.path.join(args.outdir,'scree.png'))
    plt.show()
    plt.close()

def scree_plt_selected(ev, n_factors=5):
    plt.figure(figsize=(8,6))
    plt.scatter(range(1, len(ev) + 1), ev, s=15)
    plt.plot(range(1, len(ev) + 1), ev, markersize= 10)
    plt.scatter([n_factors], ev[n_factors-1], s=15, c='red', zorder=2)
    plt.plot([n_factors, n_factors], [0, ev[0]], c='red', linestyle='--')
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalue")
    plt.grid()
    plt.savefig(os.path.join(args.outdir,'scree.png'))
    plt.close()

def sort_part(df, threshold):
    for i in range(df.shape[1]):
        cm = df.columns[i]
        if i == 0:
            rst = df[df[cm] > threshold].sort_values(by=cm, ascending=False)
            res = df[(df[cm] <= threshold)]
        else:
            rst_temp = res[res[cm] > threshold].sort_values(by=cm, ascending=False)
            rst = pd.concat([rst, rst_temp])
            res = res[(res[cm] <= threshold)]
    return rst


def radar_plt(df, labels = ['Factor1','Factor2','Factor3','Factor4','Factor5','Factor6','Factor7']):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    columns = df.columns
    columns_value = [list(df[i]) for i in columns]
    features = [f"{i+1}" for i in range(df.shape[0])]
    angles=np.linspace(0, 2*np.pi,len(features), endpoint=False)
    angles=np.concatenate((angles,[angles[0]]))
    features = np.concatenate((features,[features[0]]))
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111,polar=True)
    for i,values in enumerate(columns_value):
        values=np.concatenate((values,[values[0]]))
        ax.plot(angles, values, 'o-', markersize= 4,linewidth=1.5,label=labels[i],c=colors[i],alpha=0.8)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(angles * 180/np.pi, features, fontsize=15)
    ax.legend(bbox_to_anchor=(0.85, 0.85),frameon=False,fontsize=20)
    # ax.set_yticklabels([0.2,0.4,0.6,0.8])
    # ax.set_theta_offset(np.pi/2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir,'factor_radar.png'))
    plt.close()  


df, df_ft = removeTooManyZeros(df, args.threshold, args.start)
chi_square_value, p_value = calculate_bartlett_sphericity(df_ft)
kmo_all, kmo_model=calculate_kmo(df_ft)

f = open(os.path.join(DIR,'factor_report.txt'),'w')
f.write("########## Factor Analysis Report ###########\n")

fa = FactorAnalyzer(df_ft.shape[-1],rotation=None)
fa.fit(df_ft)
ev,v=fa.get_eigenvalues()
scree_plt_raw(ev)

n_factors = input("please enter number of selected factors: ")
n_factors = int(n_factors)
scree_plt_selected(ev, n_factors)
"""
A Scree Plot, also known as a scree test or eigenvalue plot, is a graphical representation used to assess the results of factor analysis. 
It displays the relationship between the number of factors and the amount of variance explained, aiding in the determination of how many factors should be retained. 
The plot typically shows the eigenvalues or the amount of variance explained by each factor on the y-axis, and the factor number on the x-axis. 
By examining the scree plot, analysts can identify the point where the eigenvalues level off, indicating the number of meaningful factors to retain in the analysis.
"""

fa = FactorAnalyzer(n_factors,rotation='varimax')
fa.fit(df_ft)


#  common factor variance of variable
"""
The extracted value indicates how much each variable is expressed by the common factor, and it is generally believed that greater than 0.7 indicates that the variable is well expressed by the common factor. 
It can be seen from the table that the extracted value of most variables is greater than 0.85, and the variables can be well expressed by common factors.
"""
var_commonfactor = fa.get_communalities()
f.write("Table 1. Common factor variance of variables\n")
f.write("------------------------------\n")
f.write("Feature\tVarianceCommon\n")
f.write("------------------------------\n")
for i,var in enumerate(var_commonfactor):
    f.write(df_ft.columns[i]+'\t'+str(var)+'\n')
f.write("------------------------------\n\n")

# eigenvalues
ev = fa.get_eigenvalues()


# total variance, ratia variance, cumulative variance
"""
In general, the original information loss is less, and the analysis effect of factors is ideal, which is of research significance
"""
f.write("Table 2. Variance contribution\n")
f.write("------------------------------\n")
f.write("Factor\t")
for i in range(n_factors):
    f.write("Factor{}\t".format(i))
f.write("\n")
f.write("------------------------------\n")
total_var, ratia_var, cum_var = fa.get_factor_variance()
f.write("total_var\t") 
for i in range(n_factors):
    f.write("{}\t".format(total_var[i]))
f.write("\n")    
f.write("ratia_var\t") 
for i in range(n_factors):
    f.write("{}\t".format(ratia_var[i]))
f.write("\n")   
f.write("cum_var\t") 
for i in range(n_factors):
    f.write("{}\t".format(cum_var[i]))
f.write("\n")   
f.write("------------------------------\n\n")


# factor loading
"""
"""
loading = fa.loadings_
f.write("Table 3. Component Matrix\n")
f.write("------------------------------\n")
f.write("Feature\t")
for i in range(n_factors):
    f.write("Factor{}\t".format(i))
f.write("\n")
f.write("------------------------------\n")
for i in range(df_ft.shape[1]):
    f.write("{}\t".format(df_ft.columns[i]))
    for j in loading[i]:
        f.write("{}\t".format(j))
    f.write("\n")
f.write("------------------------------\n\n")

df_loading = pd.DataFrame(loading)
df_loading.insert(0,'Feature', df_ft.columns)
df_loading.columns = ['Feature']+['Factor{}'.format(i+1) for i in range(n_factors) ]
df_loading_new =  df_loading.iloc[:,1:]
df_loading_new.index = list(df_loading['Feature'])
df_loading_new = sort_part(df_loading_new, 0.75)



df_filtered = df_loading_new[~(df_loading_new < 0.75).all(axis=1)]
filtered_ls = list(df_filtered.index)
df_filtered = pd.concat([df.iloc[:,:10],df[filtered_ls]],axis=1)
df_filtered.to_csv(os.path.join(DIR,'descriptors_filtered.csv'),header=True,index=False,sep='\t')

f.write("Table 4. Component Matrix Sorted and Filtered\n")
f.write("------------------------------\n")
f.write("Feature\t")
for i in range(n_factors):
    f.write("Factor{}\t".format(i))
f.write("\n")
f.write("------------------------------\n")
for index,row in df_loading_new.iterrows():
    f.write("{}\t".format(index))
    for j in row:
        f.write("{}\t".format(j))
    f.write("\n")
f.write("------------------------------\n\n")

# heatmap
plt.figure(figsize = (10,10))
ax = sns.heatmap(df_loading_new, annot=False, cmap="BuPu")
plt.tick_params(top=False,bottom=False,left=False,right=False)
plt.tight_layout()
plt.savefig(os.path.join(DIR,'heatmap_component_matrix.png'))
plt.close()

# radarmap
print(df_loading_new.shape)
radar_plt(df_loading_new)


# mean property_matrix)
"""

"""
df_newft = df[['Label']+list(df_loading_new.index)]
rst = df_newft.groupby('Label').mean()
rst_t = rst.T


scaler = MinMaxScaler(feature_range=(-1, 1))
normalized_data = scaler.fit_transform(rst)
# normalized_data = scaler.fit_transform(rst.T).T # row normalize
df_normalized = pd.DataFrame(normalized_data, columns=rst.columns,index=rst.index)
df_normalized = df_normalized.T

# sort by column
sorted_columns = np.argsort(-df_normalized.mean())
sorted_df = df_normalized.iloc[:, sorted_columns]


f.write("Table 5. Mean Property\n")
f.write("------------------------------\n")
f.write("Feature\t")
for i in range(sorted_df.shape[1]):
    f.write("{}\t".format(sorted_df.columns[i]))
f.write("\n")
f.write("------------------------------\n")
for index,row in sorted_df.iterrows():
    f.write("{}\t".format(index))
    for j in row:
        f.write("{}\t".format(j))
    f.write("\n")
f.write("------------------------------\n\n")

plt.figure(figsize = (10,10))
ax = sns.heatmap(sorted_df, annot=True, cmap='viridis')
plt.xlabel(None)
plt.tick_params(top=False,bottom=False,left=False,right=False)
plt.tight_layout()
plt.savefig(os.path.join(DIR,'heatmap_property_matrix.png'))
plt.close()
#############################################################


# new variable
var_new = fa.transform(df_ft)
f.write("Table 6. New Variable\n")
f.write("------------------------------\n")
f.write("Feature\t")
for i in range(n_factors):
    f.write("Factor{}\t".format(i))
f.write("\n")
f.write("------------------------------\n")
for i in range(df_ft.shape[1]):
    f.write("{}\t".format(df_ft.columns[i]))
    for j in var_new[i]:
        f.write("{}\t".format(j))
    f.write("\n")
f.write("------------------------------\n\n")


# new variable
var_new = fa.transform(df_ft)
var_new = pd.DataFrame(var_new, columns=df_loading.columns[1:],index=df_ft.index)
var_new.insert(0,'Label',list(df['Label']))
var_new = var_new.groupby('Label').mean()


plt.figure(figsize = (10,10))
ax = sns.heatmap(var_new, annot=True, cmap='BuPu',annot_kws={"size":15})
plt.tick_params(top=False,bottom=False,left=False,right=False)
plt.ylabel(None)
plt.tight_layout()
plt.savefig(os.path.join(DIR,'heatmap_score_matrix.png'))
plt.close()

f.write("Table 7. Group variable Matrix\n")
f.write("------------------------------\n")
f.write("Feature\t")
for i in range(var_new.shape[1]):
    f.write("{}\t".format(var_new.columns[i]))
f.write("\n")
f.write("------------------------------\n")
for index,row in var_new.iterrows():
    f.write("{}\t".format(index))
    for j in row:
        f.write("{}\t".format(j))
    f.write("\n")
f.write("------------------------------\n\n")




