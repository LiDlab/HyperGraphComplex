from layers import InnerProductDecoder,Attention,HGNNConv
from layers import Concentration,Classifier_MLP
import torch
import torch.nn as nn
import torch.nn.functional as F

class HGVAE(nn.Module):
    r"""The HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019)."""
    def __init__(
            self,
            in_channels: int,
            hid_channels: int,
            num_classes: int,
            num_hyperedges: int,
            use_bn: bool = False,
            drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layer1 = HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        self.layer2 = HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        self.layer3 = HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        self.attention = Attention(num_classes,50,num_hyperedges)
        self.decoder = InnerProductDecoder(drop_rate, act=lambda x: x)

    def encode(self, x, hg):
        hidden1 = self.layer1(x,hg)
        return self.layer2(hidden1,hg), self.layer3(hidden1,hg)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, X, hg):
        r"""The forward function.
        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhgHypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        mu, logvar = self.encode(X,hg)
        X = self.reparameterize(mu, logvar)
        Z,edge_w = self.attention(X,hg)
        H = self.decoder(X, Z)
        H = torch.sigmoid(H)
        return X, Z, H, mu, logvar,edge_w

class PCpredict(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_class: int,
            use_bn: bool = False,
            drop_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = drop_rate
        self.concentration = Concentration(in_channels,in_channels)
        self.classfier = Classifier_MLP(in_channels, num_class)

    def forward(self, x1,GP_info):
        Z  = self.concentration(x1, GP_info)
        out = self.classfier(Z)
        return out

class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0.):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        input = F.dropout(x, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class VGAE(nn.Module):
    """
    The self-supervised module of DeepDSI
    """
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(VGAE, self).__init__()
        self.gc1 = GCNLayer(input_feat_dim, hidden_dim1, dropout)   # F.relu
        self.gc2 = GCNLayer(hidden_dim1, hidden_dim2, dropout)    # lambda x: x
        self.gc3 = GCNLayer(hidden_dim1, hidden_dim2, dropout)
        self.act1 = nn.ReLU()

    def encode(self, x, adj):
        hidden1 = self.act1(self.gc1(x, adj))
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        adj_hat = torch.mm(z, z.t())
        return adj_hat

    def forward(self, x, adj, sigmoid: bool = True):
        mu, logstd = self.encode(x, adj)
        z = self.reparameterize(mu, logstd)
        return (torch.sigmoid(self.decode(z)), z, mu, logstd) if sigmoid else (self.decode(z), z, mu, logstd)
