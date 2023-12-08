#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import logging
import argparse
from tqdm import trange
from utils.utils import *
from models.model import KGEModel, PNA_IMC_Model
from utils.train import train_kge_step, train_pna_imc_model_step
from utils.evaluate import evaluate_kge_model, evaluate_pna_imc_model
from utils.dataloader import load_kge_data,load_graph_data, get_train_and_test_cpi


def parse_args(args=None):
    """
    Parse command-line arguments for training and testing CKG-IMC Model.

    Args:
        args (list): List of command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Training and Testing CKG-IMC Model',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--device', default='cuda:0', type=str, help='Device for training (default: cuda:0)')
    
    parser.add_argument('--do_predict', action='store_true', help='Perform prediction instead of training')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the data directory (default: ./data)')
    parser.add_argument('--save_path', default='./experiments/train', type=str, help='Path to save experiment results (default: ./experiments/train)')
    parser.add_argument('-init', '--init_checkpoint', action='store_true', help='Initialize from a pre-trained checkpoint')
    
    parser.add_argument('--ccs_threshold', default=0.68, type=float, help='Threshold for CCS matrix (default: 0.68)')
    parser.add_argument('--pps_threshold', default=0.47, type=float, help='Threshold for PPS matrix (default: 0.47)')

    parser.add_argument('--pna_ccs_threshold', default=0.64, type=float, help='Threshold for CCS in PNA-IMC (default: 0.64)')
    parser.add_argument('--pna_pps_threshold', default=0.45, type=float, help='Threshold for PPS in PNA-IMC (default: 0.45)')

    parser.add_argument('--kg_learning_rate', default=0.001, type=float, help='Learning rate for KG training (default: 0.001)')
    parser.add_argument('--kg_negative_sample_size', default=10, type=int, help='Number of negative samples for KG training (default: 10)')
    parser.add_argument('--kg_hidden_dim', default=1024, type=int, help='Hidden dimension for KG model (default: 1024)')
    parser.add_argument('--kg_batch_size', default=1000, type=int, help='Batch size for KG training (default: 1000)')
    parser.add_argument('--kg_max_steps', default=3000, type=int, help='Maximum training steps for KG (default: 3000)')
    parser.add_argument('--kg_warm_up_steps', default=1500, type=int, help='Warm-up steps for KG training (default: 1500)')
    parser.add_argument('--kg_patience', default=7, type=int, help='Patience for early stopping in KG training (default: 7)')
    parser.add_argument('--kg_valid_steps', default=50, type=int, help='Train log every xx steps in KG training (default: 50)')

    parser.add_argument('-g', '--gamma', default=12.0, type=float, help='Hyperparameter for KG model (default: 12.0)')
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true', help='Use negative adversarial sampling')
    parser.add_argument('--adversarial_temperature', default=1.0, type=float, help='Temperature for adversarial sampling (default: 1.0)')
    parser.add_argument('--regularization', default=0.0, type=float, help='L2 regularization for KG model (default: 0.0)')
    parser.add_argument('--uni_weight', action='store_true', help='Use subsampling weighting like in word2vec')

    parser.add_argument('--pna_imc_max_epochs', default=1500, type=int, help='Maximum epochs for PNA-IMC training (default: 1500)')
    parser.add_argument('--pna_imc_valid_epochs', default=50, type=int, help='Validation epochs for PNA-IMC (default: 50)')
    parser.add_argument('--pna_imc_warm_up_epochs', default=400, type=int, help='Warm-up epochs for PNA-IMC training (default: 400)')
    parser.add_argument('--pna_imc_patience', default=4, type=int, help='Patience for early stopping in PNA-IMC (default: 4)')

    parser.add_argument('--pna_hidden_dim', default=512, type=int, help='Hidden dimension for PNA-IMC (default: 512)')
    parser.add_argument('--pna_edge_dim', default=512, type=int, help='Edge dimension for PNA-IMC (default: 512)')
    parser.add_argument('--pna_n_layer', default=1, type=int, help='Number of layers for PNA-IMC (default: 1)')
    parser.add_argument('--pna_imc_learning_rate', default=0.0001, type=float, help='Learning rate for PNA-IMC (default: 0.0001)')
    parser.add_argument('--pna_imc_weight_decay', default=1e-8, type=float, help='Weight decay for PNA-IMC (default: 1e-8)')

    parser.add_argument('--imc_compound_dim', default=1024, type=int, help='Dimension for compound in IMC (default: 1024)')
    parser.add_argument('--imc_protein_dim', default=3072, type=int, help='Dimension for protein in IMC (default: 3072)')
    parser.add_argument('--imc_k', default=2048, type=int, help='Parameter k for IMC (default: 2048)')

    parser.add_argument('--fold', type=int, default=0, help='Do not manually set (used for cross-validation)')

    parser.add_argument('--n_compound', type=int, default=8360, help='Do not manually set (number of compounds)')
    parser.add_argument('--n_protein', type=int, default=1975, help='Do not manually set (number of proteins)')

    return parser.parse_args(args)


        
def main(fold,args):
    args.fold = fold
    logging.info("fold: %d"%fold)

    # get valid and test triples
    _,_,valid_triples,test_triples = get_train_and_test_cpi(args)
    
    embedding_save_path = os.path.join(args.save_path,'entity_embedding_fold_%d_dim_%d.npy'%(args.fold+1,args.kg_hidden_dim))

    # load entity embedding if exist
    if os.path.exists(embedding_save_path):
        logging.info(f'Load embedding from checkpoint and skip training')
        compound_embedding,protein_embedding = load_embedding(embedding_save_path,args)
    else:
        logging.info(f'Training KGE Model')
        # load data for KGE Model 
        kge_train_iterator, nentity, nrelation = load_kge_data(args)
        
        # Set kge training configuration
        current_learning_rate = args.kg_learning_rate
        wait, best_aupr, best_step = 0, 0.0, args.kg_max_steps 

        kge_model = KGEModel(
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=args.kg_hidden_dim,
            gamma=args.gamma
        ).to(args.device)
        
        kg_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        if args.kg_warm_up_steps:
            warm_up_steps = args.kg_warm_up_steps
        else:
            warm_up_steps = args.kg_warm_up_steps // 2

        # KGE Training Loop
        for step in trange(1,args.kg_max_steps+1):
            # train model
            train_kge_step(kge_model, kg_optimizer, kge_train_iterator, args)
            
            # If the warm-up phase concludes, the learning rate will be multiplied by 0.1.
            if step >= warm_up_steps:
                current_learning_rate /= 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            # Valid model
            if step>=args.kg_warm_up_steps and step % args.kg_valid_steps == 0:
                result = evaluate_kge_model(kge_model,valid_triples,args.fold,step,args)
                if result['aupr'] > best_aupr:
                    wait, best_step, best_aupr  = 0, step, result['aupr']
                    save_embedding(kge_model,embedding_save_path)
                else:
                    wait+=1
            
            # Early stopp
            if wait >= args.kg_patience:
                logging.info(f"Early stopping after {step} steps.")
                break
        logging.info(f'Load best embedding at step {best_step}')
        compound_embedding,protein_embedding = load_embedding(embedding_save_path,args)
        del kge_model, kge_train_iterator, kg_optimizer


    # Set PNA IMC training configuration
    logging.info(f'Training PNA IMC Model')
    # load data for PNA IMC Model 
    graph_data,CPI_train,CPI_train_mask = load_graph_data(args)
    
    model = PNA_IMC_Model(graph_data, CPI_train, CPI_train_mask, compound_embedding, protein_embedding ,args).to(args.device)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.pna_imc_learning_rate,
        weight_decay=args.pna_imc_weight_decay
    )
    wait, best_aupr = 0, 0.0
    model_save_path = os.path.join(args.save_path,'best_model.pth')

    # PNA IMC Training Loop
    for epoch in trange(1,args.pna_imc_max_epochs+1):
        train_pna_imc_model_step(model,optimizer)
        
        # Validate model
        if epoch >= args.pna_imc_warm_up_epochs and epoch % args.pna_imc_valid_epochs == 0:
            logging.info("Valid model")
            valid_result = evaluate_pna_imc_model(model,valid_triples,args.fold,epoch)
            if valid_result['aupr']>best_aupr:
                best_aupr = valid_result['aupr']
                torch.save({'model': model.state_dict(),'epoch': epoch}, model_save_path)
                wait = 0
            else:
                wait += 1
        
        # Early stopping
        if wait >= args.pna_imc_patience:
            logging.info(f"Early stopping after {epoch} epochs.")
            break

    # Loading and Testing best model
    logging.info("Loading and Testing best model")
    state_dict = torch.load(model_save_path)
    model.load_state_dict(state_dict['model'])

    # Do predict
    if args.do_predict:
        predict_top_50_cpi(model,args.data_path,args.save_path,args.n_compound)
        predict_compounds_targeting_tau_Abeta42(model,args.data_path,args.save_path,args.n_compound)
        return 
    
    # Test model
    result = evaluate_pna_imc_model(model,test_triples,args.fold,state_dict['epoch'])
    save_metrics(result,args)
    logging.info(args)

    
if __name__ == '__main__':
    
    args = parse_args()
    set_logger(args)

    if args.do_predict:
        main(fold=999,args=args)
        exit()
    
    for fold in range(10):
        main(fold,args)