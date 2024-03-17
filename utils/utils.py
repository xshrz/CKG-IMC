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
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
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

