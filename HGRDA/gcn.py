import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.W1 = Parameter(torch.FloatTensor(in_ft,128), requires_grad=True)
        torch.nn.init.xavier_normal_(self.W1)
        self.W2 = Parameter(torch.FloatTensor(128,out_ft), requires_grad=True)
        torch.nn.init.xavier_normal_(self.W2)
        torch.nn.init.xavier_normal_(self.W)
        self.act = act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)



