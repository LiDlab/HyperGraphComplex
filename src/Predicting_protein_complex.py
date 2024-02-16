import pickle
import numpy as np
import pandas as pd
import torch
from utils import is_sublist

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def read_ppi_file(file_path):
    with open(file_path, "r") as file:
        ppi_pairs = [line.strip().split() for line in file.readlines()]
    return ppi_pairs

model = torch.load('../data/Saccharomyces_cerevisiae/output/model/Mann_PPI_cliques_HGVAE_PC_DNN_5CV_model.pt')
X = torch.load('../data/Saccharomyces_cerevisiae/protein_feature/Mann_PPI_cliques_hyperedge_protein_feature_HGVAE.pt')
X =X.to(device)
model = model.to(device)


def model_score(subgraphs):
    probabilities = model(X,subgraphs)
    probabilities = probabilities.cpu().tolist()
    probabilities = [i[0] for i in probabilities]

    return probabilities


PC = pd.read_pickle('../data/Saccharomyces_cerevisiae/protein_complex/Mann_PPI_Cliques_protein_GSD_PC.pkl')
PCs = PC['edge_list']
labels = PC['labels']
index = np.where(labels == 1)[0].tolist()
PCs_1 = [PCs[i] for i in index]


f_read = open('../data/Saccharomyces_cerevisiae/Hyperedge/Mann_PPI_cliques_hyperedge_CT.pkl', 'rb')
data = pickle.load(f_read)
f_read.close()

PPI = data["PPI_edge_list"]

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

proteinlist = list(PPI_dict.keys())
# PPI Extend
def subgraph_expansion_max_one_direct_adjacent(subgraph,threshold_alpha):
    expanded_subgraph = subgraph.copy()
    adjacent_points = [PPI_dict[i] for i in expanded_subgraph]
    adjacent_points = list(set([item for sublist in adjacent_points for item in sublist]))

    while True:
        adjacent = list(set(adjacent_points).difference(set(expanded_subgraph)))

        if len(adjacent) == 0:
            break

        scores = model_score([expanded_subgraph + [v] for v in adjacent])
        max_index = np.argmax(scores)

        if scores[max_index] > threshold_alpha:
            expanded_subgraph.append(adjacent[max_index])
        else:
            break

    return expanded_subgraph


def nested_list_unique(nested_list):   #嵌套列表去重
    nested_list = [sorted(sublist) for sublist in nested_list]
    nested_list_dup = []
    for sublist in nested_list:
        if sublist not in nested_list_dup:
            nested_list_dup.append(sublist)
    return nested_list_dup

def calculate_overlap_ratio(subgraph_i, subgraph_k):

    intersection = len(set(subgraph_i).intersection(set(subgraph_k)))
    union = len(set(subgraph_i).union(set(subgraph_k)))
    #union = len(set(subgraph_k))

    overlap_ratio = intersection / union

    return overlap_ratio

def subgraph_filtration(candidate_subgraphs,scores,threshold_beta):
    # 将候选复合物和分数按照分数降序排列
    sorted_complexes = sorted(zip(candidate_subgraphs, scores), key=lambda x: x[1], reverse=True)
    # 挑选候选子图中，重叠率>threshold_beta,合并之后score> 原始得分的复合物
    filtered_PC = []

    for i in range(len(sorted_complexes)):
        candidate_subgraph_i, score_i = sorted_complexes[i]
        if len(candidate_subgraph_i) != 0:
            filtered_PC_i = []
            for k in range(i+1,len(sorted_complexes)):
                candidate_subgraph_k, score_j = sorted_complexes[k]
                overlapping_ratio = calculate_overlap_ratio(candidate_subgraph_i, candidate_subgraph_k)
                if overlapping_ratio >= threshold_beta:
                    candidate_subgraph = list(set(candidate_subgraph_i).union(set(candidate_subgraph_k)))
                    if model_score([candidate_subgraph])[0] > score_i:
                        filtered_PC_i.append(candidate_subgraph)
                    else:
                        sorted_complexes[k] = [],[]
            if len(filtered_PC_i) == 0:
                filtered_PC.append(candidate_subgraph_i)
            else:
                filtered_PC.extend(filtered_PC_i)

    return filtered_PC

protein_interaction_network = PPI_dict

expanded_subgraphs = [subgraph_expansion_max_one_direct_adjacent(PPI[i],threshold_alpha = 0.8) for i in range(len(PPI))]
print('PPI based subgraph extension is complete')
expanded_subgraphs = [sorted(i) for i in expanded_subgraphs if len(i)>2]
expanded_subgraphs_score = model_score(expanded_subgraphs)
final_subgraphs = subgraph_filtration(expanded_subgraphs,expanded_subgraphs_score, threshold_beta=0.8)

final_subgraphs_dup = []
for i in final_subgraphs:
    if  not is_sublist(i,final_subgraphs_dup):
        final_subgraphs_dup.append(i)
print('Subgraph filtering is complete')

final_subgraphs = [sorted(i) for i in final_subgraphs]
final_subgraphs_score = model_score(final_subgraphs)

score_0 = [i for i in final_subgraphs_score if i>0.8]
predict_PC = [final_subgraphs[i] for i in range(len(final_subgraphs_score)) if final_subgraphs_score[i]>0.8]

protein_list = pd.read_csv("../data/Saccharomyces_cerevisiae/Gene_Entry_ID_list/Mann_PPI_cliques_hyperedge_protein_list.csv")
protein_dict = dict(zip(protein_list['ID'],list(protein_list['Gene_symbol'])))

predict_PC = [[protein_dict[val] for val in sublist] for sublist in predict_PC]
predict_PC_ID = pd.DataFrame([';'.join(map(str, x)) for x in predict_PC])
predict_PC_ID.columns = ['predict_pc']
predict_PC_ID['score'] = score_0
predict_PC_ID.to_csv('../data/Saccharomyces_cerevisiae/Predict_PC/Predict_PC_dataset.csv',index=False)
