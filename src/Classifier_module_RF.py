import argparse
import torch
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from evaluation import calculate_fmax
from pandas.core.frame import DataFrame
from sklearn.metrics import precision_recall_curve,roc_auc_score
from sklearn.metrics import auc
import pickle

def train(args):
    # protein feature
    print('Loading protein feature ......')
    feature_path = os.path.join(args.data_path, args.species, args.feature_path,
                                "Mann_PPI_cliques_hyperedge_protein_feature_HGVAE.pt")
    X = torch.load(feature_path)
    X= X.data.cpu().numpy()

    # complex
    print('Loading protein complex GSD ......')
    GSD_PC_PATH = os.path.join(args.data_path, args.species, args.complex_path,
                               "Mann_PPI_Cliques_protein_GSD_PC.pkl")
    PC = pd.read_pickle(GSD_PC_PATH)
    PCs = PC['edge_list']
    labels = PC['labels']
    label = np.array(labels).reshape(1,-1)[0]

    # complex Feature
    PCs_X = np.array([])
    for i in range(len(PCs)):
        PC_X = np.mean(X[PCs[i]], axis=0)
        if len(PCs_X) == 0:
            PCs_X = PC_X
        else:
            PCs_X = np.vstack((PCs_X, PC_X))

    # 创建随机森林分类器模型
    rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42)

    # 再划分一下5倍交叉验证
    all_idx = list(range(len(labels)))
    np.random.shuffle(all_idx)
    rs = KFold(n_splits=5, shuffle=True)
    cv_index_set = rs.split(all_idx)  # Five-fold cross validation and independent testing

    all_Y_lable = np.array([])
    all_Y_pred = np.array([])

    for train_index, test_index in cv_index_set:
        np.random.shuffle(train_index)
        np.random.shuffle(test_index)
        train_index = train_index.tolist()
        test_index = test_index.tolist()
        Train = PCs_X[train_index]
        Test = PCs_X[test_index]
        y_training = label[train_index]
        y_pred = rf_clf.fit(Train, y_training).predict_proba(Test)
        all_Y_lable = np.concatenate((all_Y_lable, label[test_index]), axis=0)
        all_Y_pred = np.concatenate((all_Y_pred, y_pred[:, 1]), axis=0)
    print('5CV Performance:')
    F1_score, threshold, Precision, Recall, Sensitivity, Specificity, ACC = calculate_fmax(all_Y_pred, all_Y_lable)
    print(F1_score, Precision, Recall, Sensitivity, Specificity, ACC, threshold)

    precision, recall, thresholds = precision_recall_curve(all_Y_lable, all_Y_pred)
    auprc = auc(recall, precision)
    print('AUPRC', auprc)

    auroc = roc_auc_score(all_Y_lable, all_Y_pred)
    print('AUROC', auroc)
    # save RF model
    filename = os.path.join(args.data_path, args.species, args.model_path,
                               "Mann_PPI_cliques_HGVAE_PC_RF_5CV_model.pkl")
    with open(filename, 'wb') as file:
        pickle.dump(rf_clf, file)

    # ind complex
    print('Loading protein complex Ind ......')
    ind_PC_PATH = os.path.join(args.data_path, args.species, args.complex_path,
                               "Mann_PPI_Cliques_protein_ind_PC.pkl")
    PC = pd.read_pickle(ind_PC_PATH)
    PCs = PC['edge_list']
    labels = PC['labels']
    label = np.array(labels).reshape(1, -1)[0]

    # complex Feature
    PCs_X = np.array([])
    for i in range(len(PCs)):
        PC_X = np.mean(X[PCs[i]], axis=0)
        if len(PCs_X) == 0:
            PCs_X = PC_X
        else:
            PCs_X = np.vstack((PCs_X, PC_X))

    y_pred = rf_clf.predict_proba(PCs_X)
    print('Ind Test Performance:')
    F1_score, threshold, Precision, Recall, Sensitivity, Specificity, ACC = calculate_fmax(y_pred[:, 1], label)
    print(F1_score, Precision, Recall, Sensitivity, Specificity, ACC, threshold)

    precision, recall, thresholds = precision_recall_curve(label, y_pred[:, 1])
    auprc = auc(recall, precision)
    print('AUPRC:', auprc)

    auroc = roc_auc_score(label, y_pred[:, 1])
    print('AUROC:', auroc)


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


