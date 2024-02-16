import torch
import torch.nn.functional as F
import argparse
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time
import pandas as pd
from utils import *
from models import VGAE
import scipy.sparse as sp
from layers import loss_function
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(args):
    # CT feature
    print("Import embed vector")
    feature_path = os.path.join(args.data_path, args.species,args.feature_path,
                                "Mann_PPI_cliques_hyperedge_protein_feature_CT.csv")
    X = torch.FloatTensor(np.array(pd.read_csv(feature_path)))
    features = X.to(device)

    # PPI
    PPI_path = os.path.join(args.data_path, args.species,
                                "IDChange_PPI_dataset/Mann_PPI_Gene.txt")
    adj = load_ppi_network(PPI_path, features.shape[0], 0.0)
    adj_label = create_diagonal_1_array(adj)
    adj_label = torch.tensor(adj_label).float()
    adj_label = adj_label.to(device)
    adj = sp.csr_matrix(adj)

    adj_norm = preprocess_graph(adj)
    adj_norm = SparseTensor(adj_norm)
    num_nodes = adj.shape[0]
    adj_norm = adj_norm.to(device)

    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # init model and optimizer
    model = VGAE(features.shape[1], args.hidden1, args.hidden2,dropout=args.droprate)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # train model
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        recovered, z, mu, logstd = model(features, adj_norm)  # , sigmoid = False)
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

    print("Optimization Finished!")
    print("Save HGVAE Embedding......")
    VGAE_Embedding_path = os.path.join(args.data_path, args.species, args.feature_path,
                                        "Mann_PPI_cliques_hyperedge_protein_feature_VGAE.pt")
    torch.save(hidden_emb, VGAE_Embedding_path)
    print(VGAE_Embedding_path, " Saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Global parameters
    parser.add_argument('--species', type=str, default="Saccharomyces_cerevisiae", help="which species to use.")
    parser.add_argument('--data_path', type=str, default="../data", help="path storing data.")
    parser.add_argument('--feature_path', type=str, default="protein_feature", help="path save output data")

    #VGAE training parameters
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--hidden1', type=int, default=200, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=100, help="Number of units in hidden layer 2.")
    parser.add_argument('--droprate', type=float, default=0.1, help="Dropout rate (1 - keep probability).")
    parser.add_argument('--epochs', type=int, default=500, help="Number of epochs to HGVAE.")

    args = parser.parse_args()
    print(args)
    train(args)
