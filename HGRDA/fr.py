import torch
import torch.nn as nn
from gcn import GCN
import numpy as np
from torch.nn.parameter import Parameter
import copy

class HG(nn.Module):
    def __init__(self, n_in, n_h, activation,rna_num,disease_num,train_pos_edges,train_neg_edges,adj,R_train,seed,args):
        super(HG, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gcn = GCN(n_in, n_h, activation)
        self.rna_num = rna_num
        self.disease_num = disease_num
        self.E = torch.nn.Embedding(rna_num+disease_num,rna_num+disease_num, padding_idx=0)
        torch.nn.init.xavier_normal_(self.E.weight.data)
        self.train_pos_edges = train_pos_edges
        self.train_neg_edges = train_neg_edges
        self.adj = copy.deepcopy(adj)
        self.R_train = copy.deepcopy(R_train)

        self.W = Parameter(torch.FloatTensor(2*(rna_num+disease_num), 1), requires_grad=True)
        self.w1 = Parameter(torch.FloatTensor(rna_num+disease_num, rna_num+disease_num), requires_grad=True)
        self.w2 = Parameter(torch.FloatTensor(rna_num+disease_num, rna_num+disease_num), requires_grad=True)

        self.We = Parameter(torch.FloatTensor(args.hidden, args.hidden), requires_grad=True)
        torch.nn.init.xavier_normal_(self.We)

        torch.nn.init.xavier_normal_(self.W)
        torch.nn.init.xavier_normal_(self.w1)
        torch.nn.init.xavier_normal_(self.w2)

    def findindex(self,i,n,mat):
        for index in range(n):
            if mat[i][index] == max(mat[i]):
                if index != i:
                    return index
                else:
                    supmat = mat
                    supmat[i][i] = 0
                    for supindex in range(n):
                        if supmat[i][supindex] == max(supmat[i]):
                            return supindex

    def h_2(self,emb):
        embs = [emb]
        embs = torch.stack(embs, dim=1)
        emb_out = torch.mean(embs, dim=1)
        emb_out = self.l2_norm(emb_out,fin=False)
        return emb_out


    def norm(self, a):
        min_a = torch.min(a)
        max_a = torch.max(a)
        a = (a - min_a) / (max_a - min_a)
        return a

    def l2_norm(self, input, axit=1,fin = True):
        norm = torch.norm(input, 2, axit, True) + 0.1
        output = torch.div(input, norm)
        return output


    def adjsym(self):
        dense = self.adj
        D = torch.sum(dense, dim=1).float()
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        index = dense.nonzero()
        data = dense[dense >= 1e-9]
        assert len(index) == len(data)
        adjsym = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.rna_num + self.disease_num, self.rna_num + self.disease_num]))
        adjsym = adjsym.coalesce().to('cpu')
        adjsym = adjsym.to_dense()
        return adjsym


    def gat(self,adj,CS,DS,R_train):
        adj_R1 = np.c_[CS_gat, R_train]
        adj_R2 = np.c_[R_train.T, DS_gat]
        adj = np.r_[adj_R1, adj_R2]
        self.CS = CS_gat
        self.DS = DS_gat
        adj = torch.FloatTensor(adj)
        return adj

    def calloss(self,emb):
        logits = torch.tensor([])
        embs = [emb]
        embs = torch.stack(embs, dim=1)
        emb_out = torch.mean(embs, dim=1)
        emb_out = self.l2_norm(emb_out)
        rna, disease = torch.split(emb_out, [self.rna_num, self.disease_num])
        for edge in self.train_pos_edges:
            rna_emb = rna[int(edge[0])]
            disease_emb = disease[int(edge[1]) - self.rna_num]
            rna_emb = rna_emb.unsqueeze(0)
            disease_emb = disease_emb.unsqueeze(0)
            inner_pro0 = torch.mm(rna_emb,self.We)
            inner_pro = torch.mm(inner_pro0,disease_emb.t())
            inner_pro = inner_pro.squeeze(0)
            logits = torch.cat((logits, inner_pro), 0)

        for edge in self.train_neg_edges:
            rna_emb = rna[int(edge[0])]
            disease_emb = disease[int(edge[1]) - self.rna_num]
            rna_emb = rna_emb.unsqueeze(0)
            disease_emb = disease_emb.unsqueeze(0)
            inner_pro0 = torch.mm(rna_emb, self.We)
            inner_pro = torch.mm(inner_pro0, disease_emb.t())
            inner_pro = inner_pro.squeeze(0)
            logits = torch.cat((logits, inner_pro), 0)

        lbl1 = torch.ones(len(self.train_pos_edges))
        lbl0 = torch.zeros(len(self.train_neg_edges))
        label = torch.cat((lbl1, lbl0), 0)
        return logits, label


    def getadj2(self):
        adj2_R1 = np.c_[adj_L, self.R_train]
        adj2_R2 = np.c_[self.R_train.T, adj_R]
        adj2 = np.r_[adj2_R1, adj2_R2]
        adj2 = torch.FloatTensor(adj2)
        return adj2

    def forward(self, adj, CS, DS,R_train , sparse):
        adj = self.gat(adj, CS, DS, R_train)
        h_1 = self.gcn(self.E.weight, adj,self.adj2, sparse)
        logits, label = self.calloss(h_1)
        return logits,label


    def embed(self, seq, adj, CS, DS, R_train,sparse, msk):
        adj = self.gat(adj, CS, DS, R_train)
        h_1 = self.gcn(seq, adj,self.adj2, sparse)
        return h_1.detach()

