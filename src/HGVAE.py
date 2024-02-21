import time
import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from dhg import Hypergraph
from dhg.data import Cooking200
from pandas import DataFrame
from models import HGVAE
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from layers import loss_function
import pickle
import pandas as pd
import numpy as np

set_seed(2023)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(args):

    print("Import embed vector")
    feature_path = os.path.join(args.data_path, args.species,args.feature_path, "Mann_PPI_cliques_hyperedge_protein_feature_CT.csv")
    X = torch.FloatTensor(np.array(pd.read_csv(feature_path)))
    X = X.to(device)

    print("Hypergraph construction")
    Hypergraph_path = os.path.join(args.data_path, args.species, "Hyperedge/Mann_PPI_cliques_hyperedge_CT.pkl")
    f_read = open(Hypergraph_path, 'rb')
    data = pickle.load(f_read)
    f_read.close()
    G = Hypergraph(data["num_vertices"], data["PPI_cliques_list"])
    G = G.to(device)
    he = G.state_dict['raw_groups']['main']
    edges = list(he.keys())
    H_incidence = G.H.to_dense()

    pos_weight = float(H_incidence.shape[0] * H_incidence.shape[0] - H_incidence.sum()) / H_incidence.sum()
    norm = H_incidence.shape[0] * H_incidence.shape[0] / float(
        (H_incidence.shape[0] * H_incidence.shape[0] - H_incidence.sum()) * 2)

    net = HGVAE(X.shape[1], args.hidden1, args.hidden2, len(edges),use_bn=True,drop_rate=args.droprate)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    list_loss = []
    list_epoch = []

    for epoch in range(args.epochs):
        list_epoch.append(epoch)
        net.train()
        st = time.time()
        optimizer.zero_grad()
        recovered, Z, H, mu, logvar, edge_w = net(X, G)
        loss = loss_function(preds=H, labels=G.H.to_dense(),
                             mu=mu, logvar=logvar, n_nodes=data["num_vertices"],
                             norm=norm, pos_weight=pos_weight)  #
        loss.backward()
        optimizer.step()
        list_loss.append(loss.item())
        print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss.item():.5f}")
    net.eval()
    recovered, Z, H, mu, logvar, edge_w = net(X, G)

    print("Save HGVAE Embedding......")
    HGVAE_Embedding_path = os.path.join(args.data_path, args.species, args.feature_path,"Mann_PPI_cliques_hyperedge_protein_feature_HGVAE.pt")
    torch.save(recovered,HGVAE_Embedding_path)
    print(HGVAE_Embedding_path," Saved!")
    #Save the embedding vector
    # path = os.path.join(args.data_path, args.species, "output/graph_embeddings_vector.pkl")
    #
    # with open(path, 'wb') as file:
    #     pkl.dump(embeddings, file)

    #read the embedding vector

    # embeddings = pd.read_pickle(os.path.join(args.data_path, args.species + "/output/vgae/graph_embeddings_vector.pkl"))

    # np.random.seed(5959)
    # #The training dataset was before 2018.1, and the independent test dataset was after 2018.1  划分数据集
    # gsp_train_file = os.path.join(args.data_path, args.species, "networks/gsp_train.txt")
    # gsp_train = pd.read_table(gsp_train_file, delimiter="\t")
    # gsp_train = np.array(gsp_train)
    # gsp_test_file = os.path.join(args.data_path, args.species, "networks/gsp_test.txt")
    # gsp_test = pd.read_table(gsp_test_file, delimiter="\t")
    # gsp_test = np.array(gsp_test)
    # gsn_train_file = os.path.join(args.data_path, args.species, "networks/gsn_train.txt")
    # gsn_train = pd.read_table(gsn_train_file, delimiter="\t")
    # gsn_train = np.array(gsn_train)
    # gsn_test_file = os.path.join(args.data_path, args.species, "networks/gsn_test.txt")
    # gsn_test = pd.read_table(gsn_test_file, delimiter="\t")
    # gsn_test = np.array(gsn_test)
    #
    # X, Y = generate_data(embeddings, gsp_train, gsn_train, args)  #
    #
    # index = np.concatenate([gsp_train, gsn_train], axis=0)
    #
    # print("The" + str(args.folds) + "fold cross validation")
    # rs = KFold(n_splits=args.folds, shuffle=True)
    # all_idx = list(range(X.shape[0]))
    # cv_index_set = rs.split(all_idx)  # Five-fold cross validation and independent testing
    # np.random.shuffle(all_idx)
    # all_X = X[all_idx]
    # all_Y = Y[all_idx]
    # index = index[all_idx]
    # all_Y_label = []
    # all_Y_pred = []
    # all_index = []
    # fold = 1
    #
    # for train_idx, test_idx in cv_index_set:
    #     np.random.shuffle(train_idx)
    #     np.random.shuffle(test_idx)
    #     X_train = X[train_idx]
    #     X_test = X[test_idx]
    #     train_label = Y[train_idx]
    #     test_label = Y[test_idx]
    #     test_index = index[test_idx]
    #
    #     print("###################################")
    #     print("The" + str(fold) + "cross validation is underway")
    #     fold = fold + 1
    #
    #     Y_pred = train_data(X_train, train_label, X_test, args.epochs)
    #
    #     Y_pred = Y_pred.data.numpy()
    #     all_Y_label.append(test_label)
    #     all_Y_pred.append(Y_pred)
    #     all_index.append(test_index)
    #
    # Y_label = np.concatenate(all_Y_label)
    # Y_pred = np.concatenate(all_Y_pred)
    # Y_pred = Y_pred.reshape(Y_pred.shape[0], 1)
    # index_fold = np.concatenate(all_index)
    #
    # plot_roc(Y_label, Y_pred, '5-CV')
    #
    # perf = evaluate_performance(Y_label, Y_pred)
    #
    # def output_data(Y_label, Y_pred, str, perf):
    #     with open(os.path.join(args.save_path, 'Y_label_' + str + '.pkl'), 'wb') as file:
    #         pkl.dump(Y_label, file)
    #     with open(os.path.join(args.save_path, 'Y_pred_' + str + '.pkl'), 'wb') as file:
    #         pkl.dump(Y_pred, file)
    #
    #     # evaluate_performance_per(Y_label, Y_pred, index_fold)
    #     if args.save_results:
    #         with open(os.path.join(args.save_path, str + ".json"), "w+") as f:
    #             json.dump(perf, f)
    #
    # output_data(Y_label, Y_pred, 'gsd', perf)
    # # ############################################################################
    # #
    # train_model(all_X, all_Y, args.epochs)
    #
    # #############################################################################
    #
    # model = torch.load("model.pkl")
    # X_, Y_ = generate_data(embeddings, gsp_test, gsn_test, args)
    #
    # # index_ind = np.concatenate([ind, gsn_test], axis=0)
    #
    # X_ = torch.from_numpy(X_).float()
    # prediction = model(X_)
    # prediction = prediction.data.numpy()
    #
    # perf = evaluate_performance(Y_, prediction)
    #
    # plot_roc(Y_, prediction, 'independent test')
    #
    # output_data(Y_, prediction, 'independent', perf)
    #
    # ####################################################################
    # positive = np.concatenate((gsp_train, gsp_test), axis=0)
    # negative = np.concatenate((gsn_train, gsn_test), axis=0)
    # X, Y = generate_data(embeddings, positive, negative, args)
    # all_idx = list(range(X.shape[0]))
    # np.random.shuffle(all_idx)
    #
    # all_X = X[all_idx]
    # all_Y = Y[all_idx]
    # train_model(all_X, all_Y, args.epochs)
    #
    # print("Predict new DSI")
    # pred_dsi = predict_new_dsi(embeddings, gsp_train, gsp_test, args)
    # pred_dsi.to_csv(os.path.join(args.save_path, 'predict_dsi.csv'))
    #
    # print('The end')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Global parameters
    parser.add_argument('--species', type=str, default="Saccharomyces_cerevisiae", help="which species to use.")
    parser.add_argument('--data_path', type=str, default="../data", help="path storing data.")
    parser.add_argument('--feature_path', type=str, default="protein_feature", help="path save output data")

    #HGVAE training parameters
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--hidden1', type=int, default=200, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=100, help="Number of units in hidden layer 2.")
    parser.add_argument('--droprate', type=float, default=0.5, help="Dropout rate (1 - keep probability).")
    parser.add_argument('--epochs', type=int, default=200, help="Number of epochs to HGVAE.")

    args = parser.parse_args()
    print(args)
    train(args)
