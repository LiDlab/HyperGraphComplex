from utils import *
import pandas as pd
import torch

def model_score(subgraphs):
    probabilities = model(X,subgraphs)
    probabilities = probabilities.cpu().tolist()
    probabilities = [i[0] for i in probabilities]

    return probabilities

model = torch.load('HyperGraphComplex_model.pt')
X = torch.load('protein_feature_HGVAE.pt')
X =X.to(device=try_gpu())
model = model.to(device=try_gpu())

f_read = open('PPI_cliques_Hyperedge.pkl', 'rb')
data = pickle.load(f_read)
f_read.close()

PPI = data["PPI_edge_list"]

PPI_dict = convert_ppi(PPI)
proteinlist = list(PPI_dict.keys())
protein_interaction_network = PPI_dict

expanded_subgraphs = [subgraph_expansion(PPI[i],threshold_alpha = 0.8) for i in range(len(PPI))]
print('PPI based subgraph extension is complete.......')
expanded_subgraphs = [sorted(i) for i in expanded_subgraphs if len(i)>2]
expanded_subgraphs_score = model_score(expanded_subgraphs)
predict_PC = subgraph_filtration(expanded_subgraphs,expanded_subgraphs_score, threshold_beta=0.8)

final_subgraphs_dup = []
for i in predict_PC:
    if  not is_sublist(i,final_subgraphs_dup):
        final_subgraphs_dup.append(i)
print('Subgraph filtering is complete')

predict_PC = [sorted(i) for i in predict_PC]
predict_PC_score = model_score(predict_PC)


protein_list = pd.read_csv("../data/Saccharomyces_cerevisiae/Gene_Entry_ID_list/Protein_list.csv")
protein_dict = dict(zip(protein_list['ID'],list(protein_list['Gene_symbol'])))

predict_PC = [[protein_dict[val] for val in sublist] for sublist in predict_PC]
predict_PC_ID = pd.DataFrame([';'.join(map(str, x)) for x in predict_PC])
predict_PC_ID.columns = ['predict_pc']
predict_PC_ID['score'] = predict_PC_score
predict_PC_ID.to_csv('../data/Saccharomyces_cerevisiae/Predict_PC/Predict_PC_dataset.csv',index=False)
