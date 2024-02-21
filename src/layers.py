import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg.structure.hypergraphs import Hypergraph
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class HGNNConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:

        # Args:
        #     X (``torch.Tensor``):
        #     hg (``dhg.Hypergraph``)
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        X = hg.smoothing_with_HGNN(X)
        if not self.is_last:
            X = self.drop(self.act(X))

        return X

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, drop_rate, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.drop_rate = drop_rate
        self.act = act

    def forward(self, X, Z):
        X = F.dropout(X, self.drop_rate, training=self.training)
        Z = F.dropout(Z, self.drop_rate, training=self.training)
        H = self.act(torch.mm(X, Z.T))
        return H

class projection(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        hidden_size: int,
    ):
        super(projection, self).__init__()
        self.linear1 = nn.Linear(hidden_channels, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        hidden_size: int,
        num_hyperedges: int,
    ):
        super(Attention, self).__init__()
        self.projections = nn.ModuleList()
        for j in range(num_hyperedges):
            self.projections.append(
            projection(hidden_channels, hidden_size)
        )

    def forward(self, X, hg):
        global Z
        he = hg.state_dict['raw_groups']['main']
        edges = list(he.keys())
        edge_w = []
        for i, projection in enumerate(self.projections):

            index = torch.tensor(list(edges[i][0]))
            index = index.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            x_he = torch.index_select(X, 0, index)
            w = projection(x_he)
            beta = torch.softmax(w, dim=0)
            edge_w.append(beta)
            #beta = beta.expand((x_he.shape[0],) + beta.shape)
            z_batch = (beta * x_he).sum(0)
            z_batch = F.leaky_relu(z_batch)     #添加激活函数
            z_batch = z_batch.unsqueeze(0)
            if (i > 0):
                Z = torch.cat([Z, z_batch], 0)
            else:
                Z = z_batch
        return torch.tanh(Z), edge_w


'''
def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)#

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

'''
def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    # cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight =pos_weight)
    cost = norm * F.binary_cross_entropy(preds, labels)

    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


class Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(in_features, 75)
        self.bn1 = nn.BatchNorm1d(75)  #BatchNorm1d-->LayerNorm
        self.fc2 = nn.Linear(75, 50)
        self.bn2 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, 25)
        self.bn3 = nn.BatchNorm1d(25)
        self.fc4 = nn.Linear(25, out_features)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):

        x = self.dropout(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.leaky_relu(self.bn3(self.fc3(x))))
        x = torch.sigmoid(self.fc4(x))

        return x
'''
#二分类模型
class Classifier(nn.Module):
    def __init__(self, in_features, num_class=2):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 75)
        self.bn1 = nn.BatchNorm1d(75)  #BatchNorm1d-->LayerNorm
        self.fc2 = nn.Linear(75, 25)
        self.bn2 = nn.BatchNorm1d(25)
        self.fc3 = nn.Linear(25, num_class)
        self.dropout = nn.Dropout(p=0.31)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        #z = torch.softmax(self.fc3(z))
        x = self.fc3(x)
        return torch.sigmoid(x)
'''
class Attention_PC(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        hidden_size: int,
        #num_drugpairs: int,
    ):
        super(Attention_PC, self).__init__()
        self.projections = projection(hidden_channels, hidden_size)

    def forward(self, x1, x2): #X是超图卷积后的特征 按事先预设好的顺序排列的药物特征
        edge_w=[]
        for i in range(x1.shape[0]):
            x_he1 =x1[i,:]
            x_he2 =x2[i,:]
            x_he1 = x_he1.unsqueeze(0)
            x_he2 = x_he2.unsqueeze(0)
            x_he = torch.cat([x_he1, x_he2], 0)
            #print(x_he1.shape,x_he2.shape,x_he.shape)
            w = self.projections(x_he)
            beta = torch.softmax(w, dim=0)
            edge_w.append(beta)
            #beta = beta.expand((x_he.shape[0],) + beta.shape)
            #z_batch = (beta * x_he).sum(0)#每对药物对的向量
            z_batch = (beta * x_he)
            z_batch = z_batch.reshape(1,-1)
            z_batch = z_batch.unsqueeze(0)
            if (i > 0):
                Z = torch.cat([Z, z_batch], 0)
            else:
                Z = z_batch
        return torch.tanh(Z),edge_w

class Concentration(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        hidden_size: int,
    ):
        super(Concentration, self).__init__()

    def forward(self, X, GP_info): #X是超图卷积后的特征 按事先预设好的顺序排列的药物特征
        edges = GP_info
        Z_bath= torch.tensor(())
        Z_bath = Z_bath.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        for i in range(len(edges)):
            Z = torch.mean(X[edges[i]], dim=0).reshape(1,-1)
            Z_bath = torch.cat((Z_bath,Z),0)
        return Z_bath

#蛋白质复合物分类
class Classifier_MLP(nn.Module):
    def __init__(
            self,
            in_features,
            out_features
    ):
        super(Classifier_MLP, self).__init__()


        self.fc1 = nn.Linear(in_features, 100)
        self.bn1 = nn.BatchNorm1d(100)

        self.fc2 = nn.Linear(100, 50)
        self.bn2 = nn.BatchNorm1d(50)

        self.fc3 = nn.Linear(50, 25)
        self.bn3 = nn.BatchNorm1d(25)

        self.fc5 = nn.Linear(25, out_features)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):

        x = self.dropout(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.sigmoid(self.fc5(x))

        return x
