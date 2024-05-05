from utils import *
from sklearn.model_selection import KFold
from evaluation import calculate_fmax,get_score
from models import PCpredict
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve,roc_auc_score
from sklearn.metrics import auc

def train_DNN(X, Y_train, Train_PC, Test_PC, epochs, learning_rate, drop_rate):
    model = PCpredict(int(X.shape[1]), 1).to(device=try_gpu())
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    Y_train = Y_train.to(device=try_gpu())
    list_epoch = []
    for e in range(epochs):
        list_epoch.append(e + 1)
        model.train()
        out = model(X, Train_PC)
        loss_train = loss(out, Y_train)
        optimizer.zero_grad()
        loss_train.backward(retain_graph=True)
        optimizer.step()

        if e % 100 == 0:
            print(f"Epoch: {e + 1}; train_loss: {loss_train.data:.4f};")
    model.eval()
    out_valid = model(X, Test_PC)
    return out_valid, model

def HGC_DNN(PC,protein_dict,PPI_dict,X):
    # id转换--AdaPPI的金标准复合物数据集
    PCs = []
    for pc in PC:
        if len(pc) > 2:  # 至少3个元素在里面才要
            if set(pc).issubset(set(list(protein_dict.keys()))):
                pc_map = [protein_dict[sub] for sub in pc]
                PCs.append(pc_map)
    PC = [sorted(i) for i in PCs]

    label1 = torch.ones(len(PC), 1, dtype=torch.float)
    all_Test_PC = []
    all_Y_lable = []
    all_Y_pred = []
    # Five-Fold cross validation
    all_idx = list(range(len(label1)))
    np.random.shuffle(all_idx)
    rs = KFold(n_splits=5, shuffle=True)
    cv_index_set = rs.split(all_idx)

    for train_index, test_index in cv_index_set:
        n=1
        print('This is the', n, 'Fold')
        n += 1
        np.random.shuffle(train_index)
        np.random.shuffle(test_index)
        train_index = train_index.tolist()
        test_index = test_index.tolist()
        # 训练集
        Train_PC = [PC[i] for i in train_index]
        Train_label1 = torch.ones(len(Train_PC), 1, dtype=torch.float)
        Train_PC_negative = negative_on_distribution(Train_PC, list(PPI_dict.keys()), 5)
        Train_label0 = torch.zeros(len(Train_PC_negative), 1, dtype=torch.float)
        Train_labels = torch.cat((Train_label1, Train_label0), dim=0)
        Train_PC_PN = Train_PC + Train_PC_negative
        all_idx = list(range(len(Train_PC_PN)))
        np.random.shuffle(all_idx)
        Train_PC_PN = [Train_PC_PN[i] for i in all_idx]
        Train_labels = Train_labels[all_idx]
        # 测试集
        Test_PC = [PC[i] for i in test_index]
        Test_label1 = torch.ones(len(Test_PC), 1, dtype=torch.float)
        Test_PC_negative = negative_on_distribution(Test_PC, list(PPI_dict.keys()), 5)
        Test_label0 = torch.zeros(len(Test_PC_negative), 1, dtype=torch.float)
        Test_labels = torch.cat((Test_label1, Test_label0), dim=0)
        Test_PC_PN = Test_PC + Test_PC_negative
        all_idx = list(range(len(Test_PC_PN)))
        np.random.shuffle(all_idx)
        Test_PC_PN = [Test_PC_PN[i] for i in all_idx]
        Test_labels = Test_labels[all_idx]

        # 训练模型
        y_pred, model = train_DNN(X, Train_labels, Train_PC_PN, Test_PC_PN, 500, 0.001, 0.3)

        all_Y_lable.append(Test_labels)
        all_Y_pred.append(y_pred)
        all_Test_PC.append(Test_PC_PN)

    all_Test_PC = [item for sublist in all_Test_PC for item in sublist]
    Y_label = [tensor.data.cpu().numpy() for tensor in all_Y_lable]
    Y_label = [item for sublist in Y_label for item in sublist]
    Y_label = np.array(Y_label)

    Y_pred = [tensor.data.cpu().numpy() for tensor in all_Y_pred]
    Y_pred = [item for sublist in Y_pred for item in sublist]
    Y_pred = np.array(Y_pred)

    F1_score_5CV, threshold, Precision, Recall, Sensitivity, Specificity, ACC = calculate_fmax(Y_pred, Y_label)
    precision, recall, thresholds = precision_recall_curve(Y_label, Y_pred)
    auprc_5CV = auc(recall, precision)
    print('AUPRC:', auprc_5CV)

    auroc_5CV = roc_auc_score(Y_label, Y_pred)
    print('AUROC:', auroc_5CV)

    predict_pc = [all_Test_PC[i] for i in range(len(Y_pred)) if Y_pred[i] > threshold]
    precision, recall, f1, acc, sn, PPV, score = get_score(PC, predict_pc)
    print(score)
    performance = [F1_score_5CV, Precision, Recall, Sensitivity, Specificity, ACC, threshold, auprc_5CV, auroc_5CV,
                   precision, recall, f1, acc, sn, PPV]

    return score
