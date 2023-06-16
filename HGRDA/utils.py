import copy
import itertools
import random
import networkx as nx
import numpy as np


def read(filename , rna_nodes_num , disease_nodes_num , weighted=False):
    G_train = nx.Graph()
    m = rna_nodes_num
    n = disease_nodes_num
    with open(filename) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                G_train.add_node(l[0], bipartite=0)

    with open(filename) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                G_train.add_node(l[1], bipartite=1)
                G_train.add_edge(l[0], l[1])

    node_list = list(G_train.nodes)
    adj_CS = np.zeros((m,m))
    adj_DS = np.zeros((n,n))
    CS = rna_sim
    DS = disease_sim
    adj_R1 = np.c_[adj_CS,R_train]
    adj_R2 = np.c_[R_train.T,adj_DS]
    adj = np.r_[adj_R1,adj_R2]
    return (adj,node_list,G_train,m,n,CS,DS,R_train,association_matrix)


def split(input_edgelist, seed, testing_ratio, weighted=False):
    G = nx.Graph()
    rna_nodes_num = 0
    disease_nodes_num = 0
    with open(input_edgelist) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                if l[0] not in G.nodes:
                    rna_nodes_num = rna_nodes_num + 1
                G.add_node(l[0], bipartite=0)

    with open(input_edgelist) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                if l[1] not in G.nodes:
                    disease_nodes_num = disease_nodes_num + 1
                G.add_node(l[1], bipartite=1)
                G.add_edge(l[0], l[1])

    testing_edges_num = int(len(G.edges) * testing_ratio)

    testing_pos_edges = random.sample(G.edges, testing_edges_num)
    G_train = copy.deepcopy(G)
    for edge in testing_pos_edges:
        node_u, node_v = edge
        G_train.remove_edge(node_u, node_v)

    node_num2, edge_num2 = len(G_train.nodes), len(G_train.edges)
    train_graph_filename = 'graph_train.txt'
    if weighted:
        nx.write_edgelist(G_train, train_graph_filename, data=['weight'])
    else:
        nx.write_edgelist(G_train, train_graph_filename, data=False)

    return G, G_train, testing_pos_edges, train_graph_filename, rna_nodes_num, disease_nodes_num



def generate_neg_edges(original_graph, testing_edges_num,n1,n2,R_all,seed):
    L = list(original_graph.nodes())
    G = nx.Graph()
    G.add_nodes_from(L)
    G.add_edges_from(itertools.combinations(L, 2))
    G.remove_edges_from(original_graph.edges())
    random.seed(seed)
    neg_edges = random.sample(G.edges, testing_edges_num)
    return neg_edges



def load(embedding_file_name):
    with open(embedding_file_name) as f:
        node_num, emb_size = f.readline().split()
        embedding_look_up = {}
        for line in f:
            vec = line.strip().split()
            node_id = vec[0]
            embeddings = vec[1:]
            emb = [float(x) for x in embeddings]
            emb = emb / np.linalg.norm(emb)
            emb[np.isnan(emb)] = 0
            embedding_look_up[node_id] = list(emb)
        assert int(node_num) == len(embedding_look_up)
        f.close()
    return embedding_look_up


