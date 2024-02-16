import random
import pandas as pd
import numpy as np
import torch
from utils import find_amino_acid,CT,is_sublist,load_txt_list,count_unique_elements
import pickle
from pandas.core.frame import DataFrame
import networkx as nx
import re

#导入GENE---ENTRY的蛋白质信息
gene_entry_seq = pd.read_csv('../data/Saccharomyces_cerevisiae/protein_feature/uniprot-sequences-2023.05.10-01.31.31.11.tsv',sep='\t')

ambiguous_index = gene_entry_seq.loc[gene_entry_seq['Sequence'].apply(find_amino_acid)].index
gene_entry_seq.drop(ambiguous_index, axis=0, inplace=True)
gene_entry_seq.index = range(len(gene_entry_seq))
print("after filtering:", gene_entry_seq.shape)
print("encode amino acid sequence using CT...")
CT_list = []
for seq in gene_entry_seq['Sequence'].values:
    CT_list.append(CT(seq))
gene_entry_seq['features_seq'] = CT_list

#1 导入蛋白质PPI网络
#AdaPPI---BIOGRID/DIP---PPI---预处理
#PPI = pd.read_csv('../data/Saccharomyces_cerevisiae/PPI/AdaPPI_Dataset/DIP/dip.txt', sep="\t", header=None)
#PPI_trans = PPI[[1,0]]

#nature----预处理
nature = pd.read_csv('../data/Saccharomyces_cerevisiae/PPI/Mann_PPI.csv',sep = ";")
nature['target'] = nature['target'].str.split(';')
nature = nature.explode('target')
nature = nature.reset_index(drop=True)
PPI = nature[['source','target']]
PPI = PPI[PPI['source']!=PPI['target']]#去掉自相互作用
PPI_trans = PPI[['target','source']]


#转换为蛋白Entry
PPI.columns = ['protein1','protein2']
PPI_trans.columns = ['protein1','protein2']
PPI = pd.concat([PPI,PPI_trans],axis=0)
PPI = PPI.reset_index()
PPI_protein_list = list(set(PPI['protein1'].unique()).union(set(PPI['protein2'].unique())))#PPI包含的蛋白
PPI_protein = DataFrame(PPI_protein_list)
PPI_protein['Entry'] = PPI_protein[0].apply(lambda x: gene_entry_seq[gene_entry_seq['Gene Names'].str.contains(x,case=False,na=False)]['Entry'].values[0] if gene_entry_seq['Gene Names'].str.contains(x,case = False).any() else 'NA')
PPI_protein.columns = ['Gene_symbol','Entry']
PPI_protein = PPI_protein[PPI_protein['Entry'] != 'NA']
#2.1有特征的蛋白质对应PPI蛋白
PPI_protein = PPI_protein[PPI_protein['Entry'].isin(set(gene_entry_seq['Entry']))]
'''
#AdaPPI--BIOGRID/DIP---PPI---不同的基因对应同一个Entry
#若去除列A中重复的行并保留第一次出现的行 -----不同的基因对应同一个Entry
PPI_protein = PPI_protein.drop_duplicates(subset='Entry', keep='first')
'''
PPI_protein_list = list(set(PPI_protein['Gene_symbol'].unique()))
PPI = PPI.drop(['index'],axis=1)
PPI = PPI[PPI['protein1'].isin(PPI_protein_list)]
PPI = PPI[PPI['protein2'].isin(PPI_protein_list)]

PPI_protein_list = list(set(PPI['protein1'].unique()).union(set(PPI['protein2'].unique())))

PPI_protein = PPI_protein[PPI_protein['Gene_symbol'].isin(PPI_protein_list)]
PPI_protein = PPI_protein.sort_values(by = ['Gene_symbol'])
PPI_protein_list = list(PPI_protein['Gene_symbol'].unique())
#基于PPI构建超边
protein_dict = dict(zip(PPI_protein_list,list(range(0,len(PPI_protein_list)))))
#PPI超边
PPI['protein1'] = PPI['protein1'].apply(lambda x:protein_dict[x])
PPI['protein2'] = PPI['protein2'].apply(lambda x:protein_dict[x])
PPI_hyperedge = PPI.values.tolist()
#去重
PPI_hyperedge = [sorted(sublist) for sublist in PPI_hyperedge]
PPI_hyperedge_dup = []
for sublist in PPI_hyperedge:
    if sublist not in PPI_hyperedge_dup:
        PPI_hyperedge_dup.append(sublist)

#使用Clique算法搜索最大子图
G = nx.Graph()
G.add_edges_from(PPI_hyperedge_dup)
cliques = list(nx.find_cliques(G))

unique_elements = count_unique_elements(cliques)
#2.6 超边写入pkl文件保存
edge_list_data =  {}
edge_list_data["num_vertices"] = len(unique_elements)
edge_list_data["PPI_edge_list"] = PPI_hyperedge_dup
edge_list_data["PPI_cliques_list"] = cliques

f_save = open('../data/Saccharomyces_cerevisiae/Hyperedge/Mann_PPI_cliques_hyperedge_CT.pkl', 'wb')
pickle.dump(edge_list_data, f_save)
f_save.close()
#2.7 protein_list to csv
PPI_protein['ID'] = PPI_protein['Gene_symbol'].apply(lambda x:protein_dict[x])
PPI_protein = PPI_protein.sort_values(by = ['ID'])
PPI_protein.to_csv("../data/Saccharomyces_cerevisiae/Gene_Entry_ID_list/Mann_PPI_cliques_hyperedge_protein_list.csv",index=False)
#蛋白质特征
#2.8 蛋白的特征
df = pd.merge(PPI_protein,gene_entry_seq,how='inner')
df = df.sort_values(by = ['ID'])
seq_feature_matrix=DataFrame(df['features_seq'].to_list())##序列特征转换为数据框
seq_feature_matrix.to_csv("../data/Saccharomyces_cerevisiae/protein_feature/Mann_PPI_cliques_hyperedge_protein_feature_CT.csv",index=False)

#转换ID后的PPI网络
PPI.to_csv("../data/Saccharomyces_cerevisiae/IDChange_PPI_dataset/Mann_PPI_Gene.txt", index=False, header=False, sep="\t")

#金标准复合物数据集
###AdaPPI中的金标准复合物
PC = load_txt_list('../data/Saccharomyces_cerevisiae/PPI/AdaPPI_Dataset','/golden_standard.txt')

###id
protein_list = pd.read_csv("../data/Saccharomyces_cerevisiae/Gene_Entry_ID_list/Mann_PPI_cliques_hyperedge_protein_list.csv")
protein_dict = dict(zip(protein_list['Gene_symbol'],list(protein_list['ID'])))
Entry_Dict = dict(zip(protein_list['Entry'],list(protein_list['ID'])))
Entry_list = protein_list['Entry'].values.tolist()
protein_list = protein_list['Gene_symbol'].values.tolist()

#id转换--AdaPPI的金标准复合物数据集
PC_map =[]
for pc in PC:
    if len(pc) > 2:  # 至少3个元素在里面才要
        if set(pc).issubset(set(protein_list)):
            pc_map = [protein_dict[sub] for sub in pc]
            PC_map.append(pc_map)
PC_map = [sorted(i) for i in PC_map]


#Complex Portal 数据库的酵母菌的蛋白质复合物数据集
complex_portal_PC =pd.read_csv('../data/Saccharomyces_cerevisiae/protein_complex/Complex_Portal_PC/559292.tsv',sep='\t')
#与处理一下数据
for i in range(len(complex_portal_PC)):
    pattern = r'CPX-\d+\(\d+\)'
    text = complex_portal_PC.loc[i,'Identifiers (and stoichiometry) of molecules in complex']
    if re.search(pattern,text):
        matches = re.findall(pattern,text)
        for match in matches:
            pattern3 = r'CPX-\d+'
            m = re.findall(pattern3, match)[0]
            sub = complex_portal_PC[complex_portal_PC['#Complex ac'] == m]['Identifiers (and stoichiometry) of molecules in complex'].iloc[0]
            text = text.replace(match, sub)
    complex_portal_PC.loc[i,'subnit'] = text

ind_PC = complex_portal_PC[['#Complex ac','Recommended name','subnit']]
ind_PC_pro = ind_PC['subnit'].str.split('\\|', expand=True)
ind_PC_pro = pd.concat([ind_PC_pro[col].str.replace('\\(.*?\\)', '',regex=True) for col in ind_PC_pro], axis=1)
ind_PC_pro = ind_PC_pro.stack()#行转列
ind_PC_pro = ind_PC_pro.reset_index(level=1, drop=False)
ind_PC_pro.columns = ['index','subunits']
ind_PC_new = ind_PC.join(ind_PC_pro)
ind_PC_new.reset_index(inplace=True)#重命名索引
#去除不在Entry_list的复合物
ind_PC_new = ind_PC_new[~ind_PC_new['level_0'].isin(ind_PC_new[~ind_PC_new['subunits'].isin(Entry_list)]['level_0'].unique().tolist())]#复合体中有序列特征的蛋白质
ind_PC_new['ID'] = ind_PC_new['subunits'].apply(lambda x: Entry_Dict[x])
#删掉元素小于3的复合物
value_counts = ind_PC_new['level_0'].value_counts()  # 统计每个元素的出现次数
filtered_values = value_counts[value_counts >= 3].index  # 筛选出出现次数大于等于3的元素
ind_PC_new = ind_PC_new[ind_PC_new['level_0'].isin(filtered_values)]  # 根据筛选条件保留符合要求的行

#转换为list
edge_dict = {}
edge_re = {}
for i in ind_PC_new['level_0'].unique():
    edge = np.sort(ind_PC_new[ind_PC_new['level_0'] == i]['ID'].tolist()).tolist()
    if is_sublist(edge, list(edge_dict.values())):
        edge_re[i] = edge
    else:
        edge_dict[i] = edge

edge_list = list(edge_dict.values())

#删除独立测试集中与PC_map重复的复合物
ind_edge_dict ={}
ind_edge_re = {}
for i in edge_dict.keys():
    if is_sublist(edge_dict[i], PC_map):
        ind_edge_re[i] = edge_dict[i]
    else:
        ind_edge_dict[i] = edge_dict[i]

ind_edge_list = list(ind_edge_dict.values())

###构建阴性数据集
#导入PPI网络
f_read = open('../data/Saccharomyces_cerevisiae/Hyperedge/Mann_PPI_cliques_hyperedge_CT.pkl', 'rb')#
data = pickle.load(f_read)
f_read.close()

PPI = data["PPI_edge_list"]

#PPI转换为字典
def convert_ppi(ppi_list):
    ppi_dict = {}
    for ppi in ppi_list:
        for protein in ppi:
            if protein not in ppi_dict:
                ppi_dict[protein] = []
            other_proteins = [p for p in ppi if p != protein]
            ppi_dict[protein].extend(other_proteins)
    return ppi_dict

PPI_dict = convert_ppi(PPI)
#使用Clique算法搜索最大子图
G = nx.Graph()
G.add_edges_from(PPI)
cliques = list(nx.find_cliques(G))

#如果取独立测试集的阴性数据集
#PC_map = ind_edge_list

#取阴性数据集
PC_subunit_count = {}
for pc in PC_map:
    length = len(pc)
    if length in PC_subunit_count:
        PC_subunit_count[length] += 1
    else:
        PC_subunit_count[length] = 1

PC_subunit_count =  pd.DataFrame(PC_subunit_count,index=[0]).T.reset_index()
PC_subunit_count.columns = ['subunits_counts','counts_subunits_counts']

###随机采样--阴性数据集
def select_subunits(df, protein_list):
    complex_list = []
    for index, row in df.iterrows():
        subunit_count = row['subunits_counts']
        complex_count = row['counts_subunits_counts']
        for _ in range(complex_count*5):
            subunits = random.sample(protein_list, subunit_count)
            complex_list.append(subunits)
    return complex_list

PCs_N = select_subunits(PC_subunit_count,list(PPI_dict.keys()))
PCs_N = [sorted(i) for i in PCs_N]
#去掉和阳性数据集重叠的边
for i in PCs_N:
    if is_sublist(i,PC_map):
        PCs_N.remove(i)

PC_PN = PC_map+PCs_N
label1 = torch.ones(len(PC_map),1,dtype= torch.float)
label0 = torch.zeros(len(PCs_N),1,dtype= torch.float)
labels = torch.cat((label1,label0),dim=0)
all_idx = list(range(len(PC_PN)))
np.random.shuffle(all_idx)
PC_PN = [PC_PN[i] for i in all_idx]
labels = labels[all_idx]

PC = {}
PC["labels"] = labels
PC["edge_list"] = PC_PN

f_save = open('../data/Saccharomyces_cerevisiae/protein_complex/Mann_PPI_Cliques_protein_GSD_PC.pkl', 'wb')
pickle.dump(PC, f_save)
f_save.close()


#Independent test set
PC_map = ind_edge_list

#取阴性数据集
PC_subunit_count = {}
for pc in PC_map:
    length = len(pc)
    if length in PC_subunit_count:
        PC_subunit_count[length] += 1
    else:
        PC_subunit_count[length] = 1

PC_subunit_count =  pd.DataFrame(PC_subunit_count,index=[0]).T.reset_index()
PC_subunit_count.columns = ['subunits_counts','counts_subunits_counts']

PCs_N = select_subunits(PC_subunit_count,list(PPI_dict.keys()))
PCs_N = [sorted(i) for i in PCs_N]
#去掉和阳性数据集重叠的边
for i in PCs_N:
    if is_sublist(i,PC_map):
        PCs_N.remove(i)

PC_PN = PC_map+PCs_N
label1 = torch.ones(len(PC_map),1,dtype= torch.float)
label0 = torch.zeros(len(PCs_N),1,dtype= torch.float)
labels = torch.cat((label1,label0),dim=0)
all_idx = list(range(len(PC_PN)))
np.random.shuffle(all_idx)
PC_PN = [PC_PN[i] for i in all_idx]
labels = labels[all_idx]

PC = {}
PC["labels"] = labels
PC["edge_list"] = PC_PN

f_save = open('../data/Saccharomyces_cerevisiae/protein_complex/Mann_PPI_Cliques_protein_ind_PC.pkl', 'wb')
pickle.dump(PC, f_save)
f_save.close()
