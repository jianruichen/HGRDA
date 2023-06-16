import torch
import torch.nn as nn
import networkx as nx
import pandas as pd
from gcn import *
from fr import *
from utils import *

def embedding(args, train_graph_filename,G,rna_nodes_num,disease_nodes_num):
    g= read(train_graph_filename,rna_nodes_num,disease_nodes_num)

    n1n2R = training(args,G,G_=g)

    return n1n2R


def training(args,G,G_=None):
    seed=args.seed
    nb_epochs = args.epochs
    lr = 0.001
    sparse = False
    nonlinearity = 'ReLU'
    adj = G_[0]
    node_list = []
    for i in range(G_[3] + G_[4]):
        node_list.append(i)
    G_train = G_[2]
    n_train_rna = G_[3]
    n_train_disease = G_[4]
    CS = G_[5]
    DS = G_[6]
    R_train = G_[7]
    R_all = G_[8]

    adj = torch.FloatTensor(adj)
    train_neg_edges = generate_neg_edges(G, len(G_train.edges()), n_train_rna, n_train_disease,R_all, seed)
    train_pos_edges = G_train.edges()

    model = HG(n_train_rna + n_train_disease, args.hidden, nonlinearity,n_train_rna,n_train_disease,train_pos_edges,train_neg_edges,adj,R_train,seed,args)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    F = nn.Sigmoid()
    F_loss = nn.BCELoss()
    wait = 0
    best = 1e10


    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()
        logits , label = model(adj,CS,DS,R_train,sparse)
        logits = torch.FloatTensor(logits)
        logits = F(logits)


        label = torch.FloatTensor(label)

        loss = F_loss(logits, label)
        print('Loss:', loss)

        if loss < best:
            best = loss
            best_t = epoch
            wait = 0
            torch.save(model.state_dict(), 'best.pkl')
        else:
            wait += 1

        loss.backward()
        optimiser.step()
    model.load_state_dict(torch.load('best.pkl'))

    embeds = model.embed(model.E.weight , adj,CS, DS, R_train, sparse, None)

    output = args.output
    Count = embeds.numpy()

    fout = open(output, 'w')
    fout.write("{} {}\n".format(Count.shape[0], Count.shape[1]))
    for idx in range(Count.shape[0]):
        fout.write("{} {}\n".format(node_list[idx], ' '.join([str(x) for x in Count[idx, :]])))
    fout.close()
    return (R_train.shape[0] , R_train.shape[1] , R_all,train_neg_edges)




















