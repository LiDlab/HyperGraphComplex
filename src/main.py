import torch.optim as optim
from dhg import Hypergraph
from models import HGVAE
from Train_PC import HGC_DNN
import argparse
from torch.optim import Adam
import time
from utils import *
from models import VGAE
import scipy.sparse as sp
from layers import loss_function
from node2vec import Node2Vec
from utils import load_txt_list,count_unique_elements
import networkx as nx
import pandas as pd
import numpy as np
import torch
from pandas.core.frame import DataFrame

def main(args):
    # Loading Protein sequence feature #################################################################################
    Sequence_path = os.path.join(args.data_path, args.species, args.feature_path,
                                "uniprot-sequences-2023.05.10-01.31.31.11.tsv")
    Sequence = pd.read_csv(Sequence_path,sep='\t')
    Sequence_feature = sequence_CT(Sequence)
    # Loading PPI ######################################################################################################
    PPI = os.path.join(args.data_path, args.species, args.PPI_path,
                                "Mann_PPI.csv")
    PPI = pd.read_csv(PPI, sep=";")
    PPI = (
        PPI.assign(target=PPI['target'].str.split(';'))
        .explode('target')
        .reset_index(drop=True)
        [['source', 'target']]
        .query('source != target')
    )
    PPI_trans = PPI[['target', 'source']].copy()
    PPI_trans.columns = ['protein1', 'protein2']
    PPI.columns = ['protein1', 'protein2']
    PPI = pd.concat([PPI, PPI_trans], axis=0).reset_index(drop=True)

    PPI,Protein_dict = preprocessing_PPI(PPI,Sequence_feature)
    PPI.to_csv(os.path.join(args.data_path, args.species, args.PPI_path,
                                "ID_Change_PPI.txt"),
               index=False, header=False,sep="\t")

    Protein_dict.to_csv(os.path.join(args.data_path, args.species,
                            "Gene_Entry_ID_list/Protein_list.csv"),
               index=False, header=False, sep="\t")

    PPI_list = PPI.values.tolist()
    PPI_list = Nested_list_dup(PPI_list)
    # Constructing PPI hypergraph ######################################################################################
    G = nx.Graph()
    G.add_edges_from(PPI_list)
    PPI_hyperedge_dup = list(nx.find_cliques(G))
    unique_elements = count_unique_elements(PPI_hyperedge_dup)

    edge_list_data = {}
    edge_list_data["num_vertices"] = len(unique_elements)
    edge_list_data["PPI_edge_list"] = PPI_list
    edge_list_data["PPI_cliques_list"] = PPI_hyperedge_dup

    f_save = open(os.path.join(args.data_path, args.species, args.PPI_path,
                                "'PPI_cliques_Hyperedge.pkl'"), 'wb')
    pickle.dump(edge_list_data, f_save)
    f_save.close()
    # PPI hypergraph -- Protein sequence ###############################################################################
    Sequence_feature = pd.merge(Protein_dict, Sequence_feature, how='inner')
    Sequence_feature = Sequence_feature.sort_values(by=['ID'])
    Sequence_feature = DataFrame(Sequence_feature['features_seq'].to_list())
    X = torch.FloatTensor(np.array(Sequence_feature))
    X = X.to(device=try_gpu())
    CT_Embedding_path = os.path.join(args.data_path, args.species, args.feature_path,
                                "protein_feature_CT.pt")
    torch.save(X, CT_Embedding_path)
    # Feature embedding ################################################################################################
    if args.model == 'HGVAE':
        G = Hypergraph(edge_list_data["num_vertices"], edge_list_data["PPI_cliques_list"])
        G = G.to(device=try_gpu())
        he = G.state_dict['raw_groups']['main']
        edges = list(he.keys())
        H_incidence = G.H.to_dense()

        pos_weight = float(H_incidence.shape[0] * H_incidence.shape[0] - H_incidence.sum()) / H_incidence.sum()
        norm = H_incidence.shape[0] * H_incidence.shape[0] / float(
            (H_incidence.shape[0] * H_incidence.shape[0] - H_incidence.sum()) * 2)

        net = HGVAE(X.shape[1], args.hidden1, args.hidden2, len(edges), use_bn=True, drop_rate=args.droprate)
        net = net.to(device=try_gpu())
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
        list_loss = []
        list_epoch = []

        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("Total number of parameters:", total_params)

        for epoch in range(args.epochs):
            list_epoch.append(epoch)
            net.train()
            st = time.time()
            optimizer.zero_grad()
            recovered, Z, H, mu, logvar, edge_w = net(X, G)
            loss = loss_function(preds=H, labels=G.H.to_dense(),
                                 mu=mu, logvar=logvar, n_nodes=edge_list_data["num_vertices"],
                                 norm=norm, pos_weight=pos_weight)  #
            loss.backward()
            optimizer.step()
            list_loss.append(loss.item())
            print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss.item():.5f}")
        net.eval()
        recovered, Z, H, mu, logvar, edge_w = net(X, G)
        Embedding = recovered
    elif args.model == 'VGAE':
        PPI_path = os.path.join(args.data_path, args.species, args.PPI_path,
                                "ID_Change_PPI.txt")
        adj = load_ppi_network(PPI_path, X.shape[0])
        adj_label = create_diagonal_1_array(adj)
        adj_label = torch.tensor(adj_label).float()
        adj_label = adj_label.to(device=try_gpu())
        adj = sp.csr_matrix(adj)

        adj_norm = preprocess_graph(adj)
        adj_norm = SparseTensor(adj_norm)
        num_nodes = adj.shape[0]
        adj_norm = adj_norm.to(device=try_gpu())

        # Create Model
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        # init model and optimizer
        model = VGAE(X.shape[1], args.hidden1, args.hidden2, dropout=args.droprate)
        model = model.to(device=try_gpu())
        optimizer = Adam(model.parameters(), lr=args.lr)

        # train model
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            recovered, z, mu, logstd = model(X, adj_norm)  # , sigmoid = False)
            loss = loss_function(preds=recovered, labels=adj_label,
                                 mu=mu, logstd=logstd,
                                 norm=norm, pos_weight=pos_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            hidden_emb = mu
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "time=",
                      "{:.5f}".format(time.time() - t))
        Embedding = hidden_emb
    elif args.model == 'Node2vec':
        graph = pd.read_csv(os.path.join(args.data_path, args.species, args.PPI_path,
                                "ID_Change_PPI.txt"), sep='\t', header=None)
        edgelist = graph.values.tolist()
        G = nx.from_edgelist(edgelist)
        model = Node2Vec(G, dimensions=64, walk_length=80, num_walks=10, p=8, q=1, workers=1)  #
        # Embed nodes
        model = model.fit(window=10, min_count=1,
                          batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
        embedding = model.wv.vectors
        embedding = pd.DataFrame(embedding)
        Embedding = torch.FloatTensor(np.array(embedding)).to(device=try_gpu())
    elif args.model == 'None':
        Embedding = X

    Embedding_path = os.path.join(args.data_path, args.species, args.feature_path,
                                     "protein_feature_HGVAE.pt")
    torch.save(Embedding, Embedding_path)

    # protein complex predict model ####################################################################################
    PPI = edge_list_data["PPI_edge_list"]
    PPI_dict = convert_ppi(PPI)
    PC = load_txt_list(os.path.join(args.data_path, args.species, args.PC_path), '/AdaPPI_golden_standard.txt')
    protein_dict = dict(zip(Protein_dict['Gene_symbol'], list(Protein_dict['ID'])))
    model_score = HGC_DNN(PC,protein_dict,PPI_dict,Embedding)
    print("model evalutation score:", model_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Global parameters
    parser.add_argument('--species', type=str, default="Saccharomyces_cerevisiae", help="which species to use.")
    parser.add_argument('--data_path', type=str, default="../data", help="path storing data.")
    parser.add_argument('--feature_path', type=str, default="protein_feature", help="feature path data")
    parser.add_argument('--PPI_path', type=str, default="PPI", help="PPI data path")
    parser.add_argument('--PC_path', type=str, default="Protein_complex", help="Protein complex data path")
    parser.add_argument('--model', type=str, default="HGVAE", help="Feature coding")

    #training parameters
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--hidden1', type=int, default=200, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=100, help="Number of units in hidden layer 2.")
    parser.add_argument('--droprate', type=float, default=0.5, help="Dropout rate (1 - keep probability).")
    parser.add_argument('--epochs', type=int, default=200, help="Number of epochs to HGVAE.")

    args = parser.parse_args()
    print(args)
    main(args)
