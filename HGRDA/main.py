import os
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import numpy as np
from embed_train import embedding
from utils import load, split




def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    
    parser.add_argument('--input', choices=[
        'CircR2Disease.txt'
        'HMDD V2.0.txt'], default='CircR2Disease.txt')
    parser.add_argument('--output', choices=[
        'Default_c.txt',
        'Default_h.txt'], default='Default_c.txt')
    parser.add_argument('--testingratio', default=0.2, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--weighted', type=bool, default=False)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--hidden', default=64, type=int)
    parser.add_argument('--seed',default=1, type=int,  help='seed value')

    args = parser.parse_args()

    return args



def main(args):
    n = 0
    AUC_ROC = []
    AUC_PR = []
    ACC = []
    F1 = []
    G, G_train, testing_pos_edges, train_graph_filename, rna_nodes_num, disease_nodes_num = split(args.input, args.seed, args.testingratio,weighted=args.weighted)
    n1n2R = embedding(args, train_graph_filename,G,rna_nodes_num, disease_nodes_num)
    embeddings = load(args.output)
    auc_roc, auc_pr, accuracy, f1 = Prediction(embeddings, G, G_train, testing_pos_edges,n1n2R[0],n1n2R[1],n1n2R[2],n1n2R[3],args.seed)
    AUC_ROC.append(auc_roc)
    AUC_PR.append(auc_pr)
    ACC.append(accuracy)
    F1.append(f1)
    AUC_ROC = np.array(AUC_ROC)
    AUC_PR = np.array(AUC_PR)
    ACC = np.array(ACC)
    F1 = np.array(F1)
    print(
            'AUC-ROC= %.4f +- %.4f | AUC-PR= %.4f +- %.4f | Acc= %.4f +- %.4f | F1= %.4f +- %.4f' % (
            AUC_ROC.mean(), AUC_ROC.std(), AUC_PR.mean(), AUC_PR.std(), ACC.mean(), ACC.std(), F1.mean(),
            F1.std()))
    return


def run_main():
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    main(parse_args())


if __name__ == "__main__":
    run_main()


