#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os

import torch
from tqdm import trange


from models.model import *
from utils.dataloader import load_kge_data,load_graph_data,get_train_and_test_cpi
from utils.evaluate import evaluate_kge_model,evaluate_pna_imc_model
from utils.train import train_kge_step,train_pna_imc_model_step
from utils.utils import *
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing CKG-IMC Model',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--device', default='cuda:0', type=str)
    
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--data_path', type=str, default='data/our_data')
    parser.add_argument('--log_path', default='./experiments/log', type=str)
    parser.add_argument('--save_path', default='./experiments/kge_embedding', type=str)
    parser.add_argument('-init','--init_checkpoint', action='store_true')
    
    parser.add_argument('--ccs_threshold', default=0.68, type=float)
    parser.add_argument('--pps_threshold', default=0.47, type=float)

    parser.add_argument('--pna_ccs_threshold', default=0.64, type=float)
    parser.add_argument('--pna_pps_threshold', default=0.45, type=float)


    # parser.add_argument('--kg_load_embedding', action='store_true')
    parser.add_argument('--kg_learning_rate', default=0.001, type=float)
    parser.add_argument('--kg_negative_sample_size', default=10, type=int)
    parser.add_argument('--kg_hidden_dim', default=1024, type=int)
    parser.add_argument('--kg_batch_size', default=1000, type=int)
    parser.add_argument('--kg_max_steps', default=3000, type=int)
    parser.add_argument('--kg_warm_up_steps', default=1500, type=int)
    parser.add_argument('--kg_patience', default=7, type=int)
    parser.add_argument('--kg_log_steps', default=50, type=int, help='train log every xx steps')

    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('--regularization', default=0.0, type=float)
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    



    parser.add_argument('--pna_imc_max_epochs', default=1500, type=int)
    # parser.add_argument('--pna_imc_log_epochs', default=100, type=int)
    parser.add_argument('--pna_imc_valid_epochs', default=50, type=int)
    parser.add_argument('--pna_imc_warm_up_epochs', default=400, type=int)
    parser.add_argument('--pna_imc_patience', default=4, type=int)

    parser.add_argument('--pna_hidden_dim', default=512, type=int)
    parser.add_argument('--pna_edge_dim', default=512, type=int)
    parser.add_argument('--pna_n_layer', default=1, type=int)
    parser.add_argument('--pna_imc_learning_rate', default=0.0001, type=float)
    parser.add_argument('--pna_imc_weight_decay', default=1e-8, type=float)

    parser.add_argument('--imc_compound_dim', default=1024, type=int)
    parser.add_argument('--imc_protein_dim', default=3072, type=int)
    parser.add_argument('--imc_k', default=2048, type=int)

    parser.add_argument('--seed', default=1024, type=int)

    parser.add_argument('--fold', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--n_compound', type=int, default=8360, help='DO NOT MANUALLY SET')
    parser.add_argument('--n_protein', type=int, default=1966, help='DO NOT MANUALLY SET')

    return parser.parse_args(args)

        
def main(fold,args):
    args.fold = fold
    for key,value in args._get_kwargs():
        logging.info("%s: %s"%(key,value))
    set_seed(args.seed)

    _,_,valid_triples,test_triples = get_train_and_test_cpi(args)
    
    embedding_save_path = os.path.join(args.save_path,'entity_embedding_fold_%d_dim_%d.npy'%(args.fold+1,args.kg_hidden_dim))

    if os.path.exists(embedding_save_path):
        compound_embedding,protein_embedding = load_embedding(embedding_save_path,args)
    else:
        
        train_iterator, nentity, nrelation = load_kge_data(args)
        
        # Set training configuration
        current_learning_rate = args.kg_learning_rate
        best_aupr, best_step, wait = 0.0, args.kg_max_steps, 0

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

        training_logs = []
        
        # KGE Training Loop
        for step in trange(1,args.kg_max_steps+1):
            log = train_kge_step(kge_model, kg_optimizer, train_iterator, args)
            
            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate /= 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step>=args.kg_warm_up_steps and step % args.kg_log_steps == 0:
                result = evaluate_kge_model(kge_model,valid_triples,args.fold,step,args)
                if result['test_aupr'] > best_aupr:
                    best_aupr, best_step, wait = result['test_aupr'], step, 0
                    save_embedding(kge_model,embedding_save_path)
                else:
                    wait+=1

            if wait >= args.kg_patience:
                logging.info(f"Early stopping after {step} steps.")
                break
        logging.info(f'Load best embedding at {best_step} step')
        compound_embedding,protein_embedding = load_embedding(embedding_save_path,args)
        # evaluate on test
        # if not args.do_predict:
        #     evaluate_kge_model(kge_model,test_triples,args.fold,best_step,args)



    # train pna_imc model
    graph_data,CPI_train,CPI_train_mask = load_graph_data(args)
    
    # model = PNA_IMC_Model(graph_data, CPI_train, CPI_train_mask, compound_embedding, protein_embedding ,args).to(args.device)
    
    # without_pna
    # model = PNA_IMC_Model_without_pna(graph_data, CPI_train, CPI_train_mask, compound_embedding, protein_embedding ,args).to(args.device)
    
    # without_kg
    # model = PNA_IMC_Model_without_kg(graph_data, CPI_train, CPI_train_mask, compound_embedding, protein_embedding ,args).to(args.device)
    
    # without_IMC
    model = PNA_IMC_Model_without_IMC(graph_data, CPI_train, CPI_train_mask, compound_embedding, protein_embedding ,args).to(args.device)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.pna_imc_learning_rate,
        weight_decay=args.pna_imc_weight_decay
    )
    model_save_path = os.path.join(args.save_path,'best_model.pth')
    best_aupr = 0.0
    metrics = []
    wait = 0
    for epoch in trange(1,args.pna_imc_max_epochs+1):
        train_pna_imc_model_step(model,optimizer)

        if epoch >= args.pna_imc_warm_up_epochs and epoch % args.pna_imc_valid_epochs == 0:
            logging.info("Valid model")
            valid_result = evaluate_pna_imc_model(model,valid_triples,args.fold,epoch)
            if valid_result['test_aupr']>best_aupr:
                best_aupr = valid_result['test_aupr']
                torch.save({'model': model.state_dict(),'epoch': epoch}, model_save_path)
                wait = 0
            else:
                wait += 1
        # 如果连续 pna_imc_patience 次迭代性能没有提升，则提前停止训练
        if wait >= args.pna_imc_patience:
            logging.info(f"Early stopping after {epoch} epochs.")
            break

    state_dict = torch.load(model_save_path)
    model.load_state_dict(state_dict['model'])
    if args.do_predict:
        predict(model,args.data_path,args.save_path,args.n_compound)
        # get_top_50_cpi(model,args.data_path,args.save_path,args.n_compound)
    else:
        logging.info("Loading and Testing best model")
        result = evaluate_pna_imc_model(model,test_triples,args.fold,state_dict['epoch'])
        metrics.append(result)
        metrics_save_path = os.path.join(args.save_path,'metrics_kg_dim_%d_pna_dim_%d_compound_dim_%d_protein_dim_%d_layer_%d.csv'%(args.kg_hidden_dim,args.pna_hidden_dim,args.imc_compound_dim,args.imc_protein_dim,args.pna_n_layer))
        save_metric(metrics,metrics_save_path)
        logging.info(args)
        return result
if __name__ == '__main__':
    
    args = parse_args()
    set_logger(args)
    save_config(args)
    if args.do_predict:
        main(fold=999,args=args)
    else:
        metrics = []
        for fold in range(1):
            result = main(fold,args)
            metrics.append(result)
        save_metric(metrics,os.path.join(args.save_path,'metrics.csv'))









    # parser.add_argument('--use_autoencoder', action='store_true')
    # parser.add_argument('--autoencoder_hidden_dim', default=512, type=int)
    # parser.add_argument('--autoencoder_compound_hidden_dim', default=512, type=int)
    # parser.add_argument('--autoencoder_protein_hidden_dim', default=512, type=int)
    # parser.add_argument('--autoencoder_learning_rate', default=0.001, type=int)
    # parser.add_argument('--autoencoder_train_epochs', default=100, type=int)
    # parser.add_argument('--autoencoder_l1_penalty', default=0.0, type=float)
    # parser.add_argument('--autoencoder_weight_decay', default=4e-7, type=float)

            # entity_embedding = kge_model.entity_embedding.detach()

        # compound_embedding = entity_embedding[:args.n_compound]
        # protein_embedding = entity_embedding[args.n_compound:args.n_compound+args.n_protein]
        
    # if args.use_autoencoder:

    #     # train entity autoencoder
    #     autoencoder_dataloader = torch.utils.data.DataLoader(entity_embedding, batch_size=2048, shuffle=False)

    #     autoencoder = SparseAutoencoder(
    #             input_dim=entity_embedding.shape[1], 
    #             hidden_dim=args.autoencoder_hidden_dim, 
    #     ).to(args.device)
        
    #     train_sparse_autoencoder(
    #         model=autoencoder, 
    #         dataloader=autoencoder_dataloader, 
    #         num_epochs=args.autoencoder_train_epochs, 
    #         learning_rate=args.autoencoder_learning_rate, 
    #         l1_penalty=args.autoencoder_l1_penalty,
    #         weight_decay=args.autoencoder_weight_decay,
    #     )
        
    #     entity_embedding = autoencoder.encoder(entity_embedding).detach()
    #     compound_embedding,protein_embedding = entity_embedding[:args.n_compound],entity_embedding[args.n_compound:args.n_compound+args.n_protein]
    # elif False:
    #     # train compound autoencoder
    #     autoencoder_dataloader = torch.utils.data.DataLoader(compound_embedding, batch_size=2048, shuffle=False)

    #     autoencoder = SparseAutoencoder(
    #             input_dim=compound_embedding.shape[1], 
    #             hidden_dim=args.autoencoder_compound_hidden_dim, 
    #     ).to(args.device)
        
    #     train_sparse_autoencoder(
    #         model=autoencoder, 
    #         dataloader=autoencoder_dataloader, 
    #         num_epochs=args.autoencoder_train_epochs, 
    #         learning_rate=args.autoencoder_learning_rate, 
    #         l1_penalty=args.autoencoder_l1_penalty,
    #         weight_decay=args.autoencoder_weight_decay,
    #     )
        
    #     compound_embedding = autoencoder.encoder(compound_embedding)
    #     # train protein autoencoder

    #     autoencoder_dataloader = torch.utils.data.DataLoader(protein_embedding, batch_size=2048, shuffle=False)

    #     autoencoder = SparseAutoencoder(
    #             input_dim=protein_embedding.shape[1], 
    #             hidden_dim=args.autoencoder_protein_hidden_dim, 
    #     ).to(args.device)
        
    #     train_sparse_autoencoder(
    #         model=autoencoder, 
    #         dataloader=autoencoder_dataloader, 
    #         num_epochs=args.autoencoder_train_epochs, 
    #         learning_rate=args.autoencoder_learning_rate, 
    #         l1_penalty=args.autoencoder_l1_penalty,
    #         weight_decay=args.autoencoder_weight_decay,
    #     )
        
    #     protein_embedding = autoencoder.encoder(protein_embedding)
    #         # entity_embedding = torch.cat([compound_embedding,protein_embedding])
    #         # save

    #     # save_kge_embedding(kge_model, embedding_save_path)

                # metrics = {}
                # for metric in training_logs[0].keys():
                #     metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                # log_metrics('Training average', step, metrics)
                # training_logs = []