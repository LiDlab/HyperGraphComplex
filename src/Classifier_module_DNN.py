from sklearn.model_selection import KFold
from evaluation import calculate_fmax
from models import PCpredict
import torch.nn as nn
import argparse
import pandas as pd
import numpy as np
import torch
import os
from pandas.core.frame import DataFrame
from datetime import datetime
from sklearn.metrics import precision_recall_curve,roc_auc_score
from sklearn.metrics import auc

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

def train_DNN(X,Y_train,Train_PC,Test_PC, epochs, learning_rate,drop_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PCpredict(int(X.shape[1]), 1).to(device)
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    list_epoch = []
    for e in range(epochs):
        list_epoch.append(e+1)
        model.train()
        out = model(X,Train_PC,drop_rate)
        loss_train = loss(out,Y_train)
        optimizer.zero_grad()
        loss_train.backward(retain_graph=True)
        optimizer.step()

        if e % 10 == 0:
            print(f"Epoch: { e + 1}; train_loss: {loss_train.data:.4f};")
    model.eval()
    out_valid = model(X,Test_PC)
    return out_valid,model

def model_score(model,X,subgraphs):
    probabilities = model(X,subgraphs)
    probabilities = probabilities.cpu().tolist()
    probabilities = [i[0] for i in probabilities]
    return probabilities

def train(args):
    # protein feature
    print('Loading protein feature ......')
    feature_path = os.path.join(args.data_path, args.species, args.feature_path,
                                "Mann_PPI_cliques_hyperedge_protein_feature_HGVAE.pt")
    X = torch.load(feature_path)
    X = X.to(device)
    #complex
    print('Loading protein complex GSD ......')
    GSD_PC_PATH = os.path.join(args.data_path, args.species, args.complex_path,
                                "Mann_PPI_Cliques_protein_GSD_PC.pkl")
    PC = pd.read_pickle(GSD_PC_PATH)
    PCs = PC['edge_list']
    labels = PC['labels']
    labels = labels.to(device)

    # Five-Fold cross validation
    all_idx = list(range(len(labels)))
    np.random.shuffle(all_idx)
    rs = KFold(n_splits=5, shuffle=True)
    cv_index_set = rs.split(all_idx)

    all_Y_lable = []
    all_Y_pred = []

    n = 1
    for train_index, test_index in cv_index_set:
        print('This is the', n,'Fold')
        np.random.shuffle(train_index)
        np.random.shuffle(test_index)
        train_index = train_index.tolist()
        test_index = test_index.tolist()
        Train_PC = [PCs[i] for i in train_index]
        Test_PC = [PCs[i] for i in test_index]
        y_pred, model = train_DNN(X, labels[train_index], Train_PC, Test_PC, args.epochs, args.lr,drop_rate=args.droprate)
        all_Y_lable.append(labels[test_index])
        all_Y_pred.append(y_pred)
        n = n + 1
    Y_pred = [tensor.data.cpu().numpy() for tensor in all_Y_pred]
    Y_pred = [item for sublist in Y_pred for item in sublist]
    Y_pred = np.array(Y_pred)

    Y_label = [tensor.data.cpu().numpy() for tensor in all_Y_lable]
    Y_label = [item for sublist in Y_label for item in sublist]
    Y_label = np.array(Y_label)

    F1_score_5CV, threshold, Precision, Recall, Sensitivity, Specificity, ACC = calculate_fmax(Y_pred, Y_label)
    print('5CV Performance:')
    print(F1_score_5CV, Precision, Recall, Sensitivity, Specificity, ACC, threshold)

    precision, recall, thresholds = precision_recall_curve(Y_label, Y_pred)
    auprc_5CV = auc(recall, precision)
    print('AUPRC:', auprc_5CV)

    auroc_5CV = roc_auc_score(Y_label, Y_pred)
    print('AUROC:', auroc_5CV)
    model_path = os.path.join(args.data_path, args.species, args.model_path,
                                       "Mann_PPI_cliques_HGVAE_PC_DNN_5CV_model.pt")
    torch.save(model, model_path)
    print(model_path,' Saved!')
    # independent tests
    # ind complex
    print('Loading protein complex Ind ......')
    ind_PC_PATH = os.path.join(args.data_path, args.species, args.complex_path,
                               "Mann_PPI_Cliques_protein_ind_PC.pkl")
    PC = pd.read_pickle(ind_PC_PATH)
    PCs = PC['edge_list']
    labels = PC['labels']

    score = model_score(model, X, PCs)
    label = labels.tolist()
    label = [item for sublist in label for item in sublist]

    score = np.array(score)
    label = np.array(label)

    print('Ind Test Performance:')
    F1_score_ind, threshold, Precision, Recall, Sensitivity, Specificity, ACC = calculate_fmax(score, label)
    print(F1_score_ind, Precision, Recall, Sensitivity, Specificity, ACC, threshold)

    precision, recall, thresholds = precision_recall_curve(label, score)
    auprc_ind = auc(recall, precision)
    print('AUPRC', auprc_ind)

    auroc_ind = roc_auc_score(label, score)
    print('AUROC', auroc_ind)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Global parameters
    parser.add_argument('--species', type=str, default="Saccharomyces_cerevisiae", help="which species to use.")
    parser.add_argument('--data_path', type=str, default="../data", help="path storing data.")
    parser.add_argument('--feature_path', type=str, default="protein_feature", help="feature path data")
    parser.add_argument('--complex_path', type=str, default="protein_complex", help="complex path data")
    parser.add_argument('--model_path', type=str, default="output/model", help="path save model data")
    #DNN training parameters
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--droprate', type=float, default=0.3, help="Dropout rate (1 - keep probability).")
    parser.add_argument('--epochs', type=int, default=500, help="Number of epochs to HGVAE.")

    args = parser.parse_args()
    print(args)
    train(args)
