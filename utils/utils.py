import os
import json
import torch
import logging
import numpy as np
from datetime import datetime
from torch_geometric.utils import degree


def save_metrics(metrics,args):
    """
    Save metrics to a CSV file.

    Args:
        metrics (dict): A dictionary containing metric values.
        args: Arguments containing information about the model and dataset.

    Returns:
        None
    """
    save_path = os.path.join(args.save_path,'metrics_kg_dim_%d_pna_dim_%d_compound_dim_%d_protein_dim_%d_layer_%d.csv'
                             %(args.kg_hidden_dim,args.pna_hidden_dim,args.imc_compound_dim,args.imc_protein_dim,args.pna_n_layer))
    if not os.path.exists(save_path):
        with open(save_path, mode='w') as f:
            f.write(','.join(map(str,metrics.keys()))+'\n')
    with open(save_path, mode='a+') as f:
        f.write(','.join(map(str,metrics.values()))+'\n')


def save_config(args):
    """
    Save model configuration to a JSON file.

    Args:
        args: Arguments containing information about the model and dataset.

    Returns:
        None
    """
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path,'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

def calc_deg(graph_data):
    """
    Calculate the in-degree histogram tensor for the given graph data.

    Args:
        graph_data: PyTorch Geometric graph data.

    Returns:
        deg (torch.Tensor): In-degree histogram tensor.
    """
    max_degree = -1
    d = degree(graph_data.edge_index[1], num_nodes=graph_data.num_nodes, dtype=torch.long)
    max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    d = degree(graph_data.edge_index[1], num_nodes=graph_data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())
    
    return deg

def adjacency_matrix_to_triples(adjacency_matrix, relation_id, threshold=0.0, row_offset=0, col_offset=0):
    """
    Convert an adjacency matrix to triples representation.

    Args:
        adjacency_matrix (numpy.ndarray): The input adjacency matrix.
        relation_id (int): Relation ID for the triples.
        threshold (float): Threshold for considering edges in the adjacency matrix.
        row_offset (int): Offset to add to row indices.
        col_offset (int): Offset to add to column indices.

    Returns:
        triples (numpy.ndarray): Triples representation of the graph.
    """
    rows, cols = np.where(adjacency_matrix > threshold)
    rows += row_offset  # Adjust row indices by adding the specified row offset
    cols += col_offset  # Adjust column indices by adding the specified column offset
    relations = np.array([relation_id]*len(rows))
    triples = np.stack([rows,relations,cols],axis=1)
    return triples

def adjacency_matrix_to_edge_index_and_weight(adjacency_matrix, threshold=0.0, row_offset=0, col_offset=0):
    """
    Convert an adjacency matrix to edge index and edge weight.

    Args:
        adjacency_matrix (numpy.ndarray): The input adjacency matrix.
        threshold (float): Threshold for considering edges in the adjacency matrix.
        row_offset (int): Offset to add to row indices.
        col_offset (int): Offset to add to column indices.

    Returns:
        edge_index (numpy.ndarray): Edge index representation of the graph.
        edge_weight (numpy.ndarray): Edge weight representation of the graph.
    """

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
    """
    Save entity embeddings from the given knowledge graph embedding (KGE) model to a specified file.

    Args:
        kge_model: The trained knowledge graph embedding model.
        embedding_save_path (str): The file path to save the entity embeddings.

    Returns:
        None
    """

    np.save(
        embedding_save_path, 
        kge_model.entity_embedding.detach().cpu().numpy()
    )

    
def load_embedding(embedding_save_path,args):
    """
    Load entity embeddings from a file and return separate embeddings for compounds and proteins.

    Args:
        embedding_save_path (str): The file path to load the entity embeddings.
        args: Arguments containing information about the dataset and device.

    Returns:
        compound_embedding: Embeddings for compounds.
        protein_embedding: Embeddings for proteins.
    """

    entity_embedding = torch.tensor(np.load(embedding_save_path)).to(args.device)
    compound_embedding = entity_embedding[:args.n_compound]
    protein_embedding = entity_embedding[args.n_compound:args.n_compound+args.n_protein]
    return compound_embedding,protein_embedding

def row_normalize(matrix):
    """
    Normalize the rows of a matrix by dividing each row by its sum.

    Args:
        matrix (numpy.ndarray): The input matrix.

    Returns:
        normalized_matrix (numpy.ndarray): The normalized matrix.
    """

    row_sums = matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    normalized_matrix = matrix / row_sums[:, np.newaxis]
    return normalized_matrix


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    log_file = os.path.join(args.save_path, 'train_%s.log'%(datetime.now().strftime("%Y_%m_%d_%H_%M")))
    save_config(args)
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
def predict_top_50_cpi(model,data_path,save_path,n_compound):
    """
    Predict the top 50 compound-protein interactions (CPIs) using the given model.

    Args:
        model: The trained model for predicting CPIs.
        data_path (str): Path to the data directory.
        save_path (str): Path to save the prediction results.
        n_compound (int): Number of compounds in the dataset.

    Returns:
        None
    """

    model.eval()
    _, CPI_reconstruct = model()
    CPI_reconstruct = CPI_reconstruct.detach().cpu().numpy()
    rows, cols = np.where(CPI_reconstruct > 0.5)
    scores = CPI_reconstruct[rows, cols]
    cpi_list = [[row, col, score] for row, col, score in zip(rows, cols, scores)]
    cpi_list.sort(key=lambda x: x[2], reverse=True)
    top_50_cpi = np.array(cpi_list[:50])
    entity_dict = np.loadtxt(os.path.join(data_path,'entities.dict'),delimiter='\t',dtype=str)[:,1]
    compound = entity_dict[top_50_cpi[:,0].astype(np.int32)]
    protein = entity_dict[top_50_cpi[:,1].astype(np.int32)+n_compound]
    top_50_cpi = np.stack([compound,protein,top_50_cpi[:,2]],axis=1)
    logging.info("Top 50 possible CPIs")
    logging.info(top_50_cpi)
    np.savetxt(os.path.join(save_path,'top_50_cpi.txt'),top_50_cpi,'%s')
    
@torch.no_grad()
def predict_compounds_targeting_tau_Abeta42(model,data_path,save_path,n_compound):
    """
    Predict compounds targeting Abeta42 and tau and save the results.

    Args:
        model: The trained model for predicting CPIs.
        data_path (str): Path to the data directory.
        save_path (str): Path to save the prediction results.
        n_compound (int): Number of compounds in the dataset.

    Returns:
        None
    """
        
    model.eval()
    _, CPI_reconstruct = model()
    CPI_reconstruct = CPI_reconstruct.detach().cpu().numpy()
    name_list = list(np.loadtxt(os.path.join(data_path,'entities.dict'),dtype=str)[:,1])
    Abeta42_idx = name_list.index('P05067') - n_compound
    TAU_idx = name_list.index('P10636') - n_compound
    Abeta42_compound_score_list = [[name_list[compound_idx],1-(1-score)**2] for compound_idx,score in enumerate(CPI_reconstruct[:,Abeta42_idx])]
    Abeta42_compound_score_list = sorted(Abeta42_compound_score_list,key=lambda x:x[1],reverse=True)
    np.savetxt(os.path.join(save_path,'Abeta42_compound_score.csv'),Abeta42_compound_score_list,'%s,%s')
    TAU_compound_score_list = [[name_list[compound_idx],1-(1-score)**2] for compound_idx,score in enumerate(CPI_reconstruct[:,TAU_idx])]
    TAU_compound_score_list = sorted(TAU_compound_score_list,key=lambda x:x[1],reverse=True)
    np.savetxt(os.path.join(save_path,'TAU_compound_score.csv'),TAU_compound_score_list,'%s,%s')
    np.save(os.path.join(save_path,'CPI_reconstruct.npy'),CPI_reconstruct)
    logging.info('Top 10 possible compounds targeting at Abeta42')
    logging.info(np.array(Abeta42_compound_score_list[:10]))
    logging.info('Top 10 possible compounds targeting at tau')
    logging.info(np.array(TAU_compound_score_list[:10]))
    logging.info("Prediction results saved at %s"%save_path)
