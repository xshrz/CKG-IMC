import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import os
from datetime import datetime
import torch
import logging
import json
from torch_geometric.utils import degree


def save_metric(metrics,save_path):
    fieldnames = metrics[0].keys()
    with open(save_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # 写入标题行
        for metric in metrics:
            writer.writerow(metric)  # 写入每一行的数据:



def save_config(args):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path,'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

def calc_deg(graph_data):
    # Compute the maximum in-degree in the training graph_data.
    max_degree = -1
    d = degree(graph_data.edge_index[1], num_nodes=graph_data.num_nodes, dtype=torch.long)
    max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    d = degree(graph_data.edge_index[1], num_nodes=graph_data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())
    
    return deg

def adjacency_matrix_to_triples(adjacency_matrix, relation_id, threshold=0.0, row_offset=0, col_offset=0):
    rows, cols = np.where(adjacency_matrix > threshold)
    rows += row_offset  # Adjust row indices by adding the specified row offset
    cols += col_offset  # Adjust column indices by adding the specified column offset
    relations = np.array([relation_id]*len(rows))
    triples = np.stack([rows,relations,cols],axis=1)
    return triples

def adjacency_matrix_to_edge_index_and_weight(adjacency_matrix, threshold=0.0, row_offset=0, col_offset=0):
    if adjacency_matrix.shape[0]==adjacency_matrix.shape[1]:
        np.fill_diagonal(adjacency_matrix, 0) # subtract_self_loop
        rows, cols = np.where(np.triu(adjacency_matrix) > threshold)
    else:
        rows, cols = np.where(adjacency_matrix > threshold)
    adjacency_matrix = row_normalize(adjacency_matrix)
    edge_weight = adjacency_matrix[rows, cols]
    rows += row_offset  # Adjust row indices by adding the specified row offset
    cols += col_offset  # Adjust column indices by adding the specified column offset
    edge_index = np.stack([rows,cols])
    return edge_index,edge_weight

def save_embedding(kge_model,embedding_save_path):
    np.save(
        embedding_save_path, 
        kge_model.entity_embedding.detach().cpu().numpy()
    )

    
def load_embedding(embedding_save_path,args):
    entity_embedding = torch.tensor(np.load(embedding_save_path)).to(args.device)
    compound_embedding = entity_embedding[:args.n_compound]
    protein_embedding = entity_embedding[args.n_compound:args.n_compound+args.n_protein]
    return compound_embedding,protein_embedding

def row_normalize(matrix):
    row_sums = matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    normalized_matrix = matrix / row_sums[:, np.newaxis]
    return normalized_matrix

def set_seed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# def log_metrics(mode, step, metrics):
#     '''
#     Print the evaluation logs
#     '''
#     for metric in metrics:
#         logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    log_file = os.path.join(args.save_path, 'train_%s.log'%(datetime.now()))
    # log_file = os.path.join(args.log_path,'train_%s.log'%(datetime.now()))
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

@torch.no_grad()
def get_top_50_cpi(model,data_path,save_path,n_compound):
    model.eval()
    _, CPI_reconstruct = model()
    CPI_reconstruct = CPI_reconstruct.detach().cpu().numpy()
    np.save(os.path.join(save_path,'CPI_reconstruction.npy'),CPI_reconstruct)
    rows, cols = np.where(CPI_reconstruct > 1.0)
    scores = CPI_reconstruct[rows, cols]
    cpi_list = [[row, col, score] for row, col, score in zip(rows, cols, scores)]
    cpi_list.sort(key=lambda x: x[2], reverse=True)
    top_50_cpi = np.array(cpi_list[:50])
    entity_dict = np.loadtxt(os.path.join(data_path,'entities.dict'),delimiter='\t',dtype=str)[:,1]
    compound = entity_dict[top_50_cpi[:,0].astype(np.int32)]
    protein = entity_dict[top_50_cpi[:,1].astype(np.int32)+n_compound]
    top_50_cpi = np.stack([compound,protein,top_50_cpi[:,2]],axis=1)
    logging.info("top_50_cpi")
    logging.info(top_50_cpi)
    np.savetxt(os.path.join(save_path,'top_50_cpi.txt'),top_50_cpi,'%s')
@torch.no_grad()
def predict(model,data_path,save_path,n_compound):
    model.eval()
    _, CPI_reconstruct = model()
    CPI_reconstruct = CPI_reconstruct.detach().cpu().numpy()
    name_list = list(np.loadtxt(os.path.join(data_path,'entities.dict'),dtype=str)[:,1])
    APP_idx = name_list.index('P05067') - n_compound
    TAU_idx = name_list.index('P10636') - n_compound
    APP_compound_score_list = [[name_list[compound_idx],1-(1-score)**2] for compound_idx,score in enumerate(CPI_reconstruct[:,APP_idx])]
    APP_compound_score_list = sorted(APP_compound_score_list,key=lambda x:x[1],reverse=True)
    np.savetxt(os.path.join(save_path,'APP_compound_score.csv'),APP_compound_score_list,'%s,%s')
    TAU_compound_score_list = [[name_list[compound_idx],1-(1-score)**2] for compound_idx,score in enumerate(CPI_reconstruct[:,TAU_idx])]
    TAU_compound_score_list = sorted(TAU_compound_score_list,key=lambda x:x[1],reverse=True)
    np.savetxt(os.path.join(save_path,'TAU_compound_score.csv'),TAU_compound_score_list,'%s,%s')
    logging.info("saved at %s"%save_path)
