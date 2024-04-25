
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
import numpy as np



base_dir = os.path.abspath(os.path.dirname(__file__))
# TODO
"""
please change smileses_filename, if_annotate and distance_base only
"""
if_annotate = False
distance_base = 25
smileses_filename = 'input.csv'
min_number_groups = 4


coordinates_filename = "tsne_coordinate.txt"
figure_save_filename = "tsne_csn.png"
csv_save_filename = "data_grouped.csv"

smileses_smiles_path = os.path.join(base_dir, smileses_filename)
smileses_coords_path = os.path.join(base_dir, coordinates_filename)
figure_save_path = os.path.join(base_dir, figure_save_filename)
csv_save_path = os.path.join(base_dir, csv_save_filename)

class Edge():
    def __init__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node
        self.similarity  = 0.
        self.distance = 0.0
        # self.dis_centrality = 0.0
        self.closeness_centrality = 0.0
    
class Node():
    def __init__(self,no,smiles):
        self.no=no
        self.smiles=smiles
        self.coord=[0.,0.]
        self.in_edges=[]
        self.out_edges=[]
        self.fp=[]
        self.k1_nodes=[]

def simle2fp(smiles):
    node_ls=[]
    for i,smile in enumerate(smiles):
       node = Node(no=i,smiles=smile)
       mol = Chem.MolFromSmiles(smile)
       fp = AllChem.GetMACCSKeysFingerprint(mol)
       node.fingerprint = fp
       node_ls.append(node)
    for nodeA in node_ls:
        for nodeB in node_ls:
            if nodeA.no != nodeB.no:
                edge = Edge(nodeA,nodeB)
                nodeA.out_edges.append(edge)
                nodeB.in_edges.append(edge)
    return node_ls

def cal_similarity(node_ls):
    for source_node in node_ls:
        for out_edge in source_node.out_edges:
            out_edge.similarity = DataStructs.DiceSimilarity(source_node.fingerprint, out_edge.to_node.fingerprint)
    return node_ls

def set_coordinate(node_ls, coordinates):
    for i,node in enumerate(node_ls):
        node_ls[i].coord = coordinates[i]
    for source_node in node_ls:
        for out_edge in source_node.out_edges:
            out_edge.distance = ((out_edge.to_node.coord[0]-source_node.coord[0])**2 + 
                                (out_edge.to_node.coord[1]-source_node.coord[1])**2)**0.5
    return node_ls


class CSN():
    """
    args:
        smiles<string>: molecule smiles format 
        coordinates<array>: shape like (2,n) where n is number of nodes
        similarity_connect<bool>: if use similarity to judge two nodes if connect
        dis_base<number>: distance cutoff reverse
        threshold<number>: similarity distance, only similarity_connect==True can be used
        if_annotate<bool>: if_annotate
    """
    def __init__(self,smiles,coordinates,similarity_connect,dis_base,threshold,if_annotate):
        self.threshold=threshold
        self.coordinates=coordinates
        self.similarity_connect = similarity_connect
        self.if_annotate = if_annotate
        self.max_x = max(coordinates[:,0])
        self.min_x = min(coordinates[:,0])
        self.max_y = max(coordinates[:,1])
        self.min_y = min(coordinates[:,1])
        range_x=max(coordinates[:,0])-min(coordinates[:,0])
        range_y=max(coordinates[:,1])-min(coordinates[:,1])
        self.range_x = range_x
        self.range_y = range_y
        self.dis_threshold=float((range_x**2+range_y**2)**0.5/dis_base)
        node_ls = simle2fp(smiles)
        node_ls = set_coordinate(node_ls, coordinates)
        node_ls = cal_similarity(node_ls)
        if not similarity_connect:
            for source_node in node_ls:
                for out_edge in source_node.out_edges:
                    if out_edge.distance < self.dis_threshold:
                        source_node.k1_nodes.append(out_edge.to_node)            
        else:
            for source_node in node_ls:
                for out_edge in source_node.out_edges:
                    if out_edge.distance < self.dis_threshold and out_edge.similarity > self.threshold:
                        source_node.k1_nodes.append(out_edge.to_node)            
        self.node_ls = node_ls
        k1nodes_ls = [node.k1_nodes for node in self.node_ls]
        k1nos_ls=[]
        for k1nodes in k1nodes_ls:
            k1nos_ls.append([node.no for node in k1nodes])
        self.k1nos_ls=k1nos_ls

    def traversal(self,no):
        lists = self.k1nos_ls
        ls = lists[no]
        old_nos=[]
        for k1 in ls:
            if k1 not in old_nos:
                try:
                    old_nos.append(k1)
                    for k2 in lists[k1]:
                        if k2 != no:
                            if k2 not in ls:
                                ls.append(k2)
                except:
                    continue
        ls.append(no)
        del old_nos
        del lists
        return ls
    
    def classify(self):
        groups=[]
        for i in range(len(self.node_ls)):
            print('traverse:',i)
            groups.append(sorted(set(self.traversal(i))))
        realgroups=[]
        for i,group in enumerate(groups) :
            if group not in realgroups:
                realgroups.append(group)
        self.groups = realgroups
        return realgroups
    
    def find_center(self):
        for group in self.groups:
            nodegroup = [self.node_ls[no] for no in group]
            for node1 in nodegroup:
                d=0
                for node2 in nodegroup:
                    d+=((node1.coord[0]-node2.coord[0])**2+(node1.coord[1]-node2.coord[1])**2)**0.5   
                # node1.dis_centrality=d/len(nodegroup)
                if len(group) >1 :
                    # repeat
                    if d==0:
                        node1.closeness_centrality = float("inf")
                    else: 
                        node1.closeness_centrality = len(nodegroup)/d
                else:
                    node1.closeness_centrality = 0
        centers=[]
        for group in self.groups:
            centgroup=[self.node_ls[no].closeness_centrality for no in group]
            centers.append(group[centgroup.index(max(centgroup))])
        return centers


    def plot(self):
        groups=self.groups
        plt.figure(figsize=(self.range_x* 1.1,self.range_y* 1.1))
        plt.xticks([])
        plt.yticks([])
        # plt.axis ('off')

        # TODO
        color_list = [ 
                    'mediumslateblue', 'gold', 'lightseagreen', 'pink',  'greenyellow', 'cornflowerblue',
                    'violet','turquoise','palegreen','red', 'mediumaquamarine',  'paleturquoise', 'powderblue',
                    ]
        
        color_dic = {
            'dark_red': '#C82423',
            'dark_blue': '#14517C',
            'orange' : '#FFA500',
            'purple': '#BEB8DC',
            'green':'#96CCCB',
            'cyan':'#96CCCB',
            'red': '#FA7F6F',
            
            'rice': '#E7DAD2', 
            
            'pink' : '#F6CAE5',
            'yellow' : '#F0E68C',
            'green': '#96C37D',
            'gray1': 'gray',
            'gray2': 'gray',
            'gray3': 'gray',
            'gray4': 'gray',
            'gray5': 'gray',
            'gray6': 'gray',
            'gray7': 'gray',
            'gray8': 'gray',
            'gray9': 'gray',
            'gray10': 'gray',
            'gray11': 'gray',
            'gray12': 'gray',
            'gray13': 'gray',
            'gray14': 'gray',
        }
        color_list = list(color_dic.values())
        color_names = list(color_dic.keys())

        gn=len(groups)
        cn=len(color_list)
        if gn>cn:
            for i in range(int(gn/cn)-1):
                color_list+=color_list
            for i in range(gn%cn):
                color_list.append(color_list[i]) 
                
        
        # plot all nodes, basic distance threshold are shown
        basic_point_size =  ((self.dis_threshold)*72)**2
        for group in groups:
            for no in group:
                node = self.node_ls[no]
                plt.scatter(node.coord[0],node.coord[1],s=basic_point_size,color='whitesmoke',alpha=1,zorder=1)
        plt.xlim(self.min_x* 1.1, self.max_x * 1.1)
        plt.ylim(self.min_y * 1.1, self.max_y * 1.1)    
        xmin, xmax, ymin, ymax = plt.axis()
        print("x_min:", xmin,"x_max:", xmax, "y_min:",ymin, "y_max",ymax)

        
        # plot group and edge(when similarity_connect=True)    
        center_point_size =  ((self.dis_threshold)*72)**2/(5)**2   
        line_width = self.dis_threshold*2
        annotate_fontsize = self.dis_threshold*10

        nums = np.array([len(group) for group in groups])
        print(nums)

        n=0
        colors = []
        for group in groups:
            # if one node one group: gray
            if len(group) < min_number_groups:
                colors.append('gray')
                for no in group:
                    node = self.node_ls[no]
                    plt.scatter(node.coord[0],node.coord[1],s=center_point_size,color='gray',zorder=3)
                    if if_annotate:
                        plt.annotate(int(node.no), xy = (node.coord[0], node.coord[1]), xytext = (node.coord[0], node.coord[1]),alpha=0.5,fontsize=annotate_fontsize)
            # > 1 node one group: color 
            else:
                colors.append(color_names[n])
                for no in group:
                    node = self.node_ls[no]
                    plt.scatter(node.coord[0],node.coord[1],s=center_point_size,color=color_list[n],zorder=3)
                    if if_annotate:
                        plt.annotate(int(node.no), xy = (node.coord[0], node.coord[1]), xytext = (node.coord[0], node.coord[1]), alpha=0.5,fontsize=annotate_fontsize)
                    # must have a edge can be connected
                    if self.similarity_connect:
                        for edge in node.out_edges:
                                if edge.similarity >= self.threshold:
                                    x1=edge.from_node.coord[0]
                                    y1=edge.from_node.coord[1]
                                    x2=edge.to_node.coord[0]
                                    y2=edge.to_node.coord[1]
                                    
                                    plt.plot([x1,x2],[y1,y2], color='#92C4DD',linewidth=line_width*edge.similarity**3,alpha=0.5,zorder=2) #'#D0CFF6'
                    # not be shown in fact, only for beauty
                    else:
                        for edge in node.out_edges:
                            if edge.distance < self.dis_threshold:
                                x1=edge.from_node.coord[0]
                                y1=edge.from_node.coord[1]
                                x2=edge.to_node.coord[0]
                                y2=edge.to_node.coord[1]
                                plt.plot([x1,x2],[y1,y2], color='#92C4DD',linewidth=line_width*edge.similarity**3,alpha=0.5,zorder=2) #'#D0CFF6'
                n+=1 
            
        plt.tight_layout(pad = 0, h_pad = None, w_pad = None, rect = None)
        plt.savefig(figure_save_path)
        plt.close()
        return colors
df = pd.read_csv(smileses_smiles_path, header=0, sep='\t')
smileses = list(df['Smiles'])
mols = [Chem.MolFromSmiles(smiles) for smiles in smileses]
fps = np.array([list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)) for mol in mols])
coords = np.loadtxt(smileses_coords_path)

csn=CSN(smileses,coords,False,distance_base,0.9,True)
print(csn.dis_threshold)
group_sorted_ids = []
group_sorted_smileses = []
group_sorted_centrality = []
group_reprenstatives = []
group_colors = []

groups = csn.classify()
reprenstatives = csn.find_center()
colors = csn.plot()

for i,group in enumerate(groups):
    
    # print('group{}: center is {}, complete group is:'.format(i,reprenstatives[i]),group)
    nodegroup = [csn.node_ls[no] for no in group]
    for node in nodegroup:
        group_sorted_ids.append(node.no)
        group_reprenstatives.append(reprenstatives[i])
        group_sorted_smileses.append(node.smiles)
        group_sorted_centrality.append(node.closeness_centrality)
        group_colors.append(colors[i])


dict = {'No': group_sorted_ids,   
        'Smiles':group_sorted_smileses,
        'Group':group_reprenstatives,
        'ClosenessCentrality':group_sorted_centrality,
        'Color':group_colors
        }
df_save = pd.DataFrame(dict)
df_save.to_csv(csv_save_path,sep="\t",index=None)
# df_save.to_excel(csv_save_path, sheet_name='Sheet1', index=False)

