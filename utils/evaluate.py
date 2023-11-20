import torch
import numpy as np
import logging
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,roc_auc_score, average_precision_score

@torch.no_grad()
def evaluate_kge_model(model,test_triples,fold, step,args):
    logging.info(f'evaluate_kge_model at step {step}')
    model.eval()
    head = model.entity_embedding[test_triples[:,0]]
    relation = model.relation_embedding[0].repeat(head.size(0),1)
    tail = model.entity_embedding[test_triples[:,1]+args.n_compound]

    re_head, im_head = torch.chunk(head, 2, dim=1)
    re_relation, im_relation = torch.chunk(relation, 2, dim=1)
    re_tail, im_tail = torch.chunk(tail, 2, dim=1)

    re_score = re_relation * re_tail + im_relation * im_tail
    im_score = re_relation * im_tail - im_relation * re_tail
    score = re_head * re_score + im_head * im_score

    score = score.sum(dim = 1)
    y_pred = score.cpu().numpy()
    return evaluate(fold,step,y_pred,test_triples[:,2])

@torch.no_grad()
def evaluate_pna_imc_model(model,test_triples,fold, epoch):
    logging.info('evaluate_pna_imc_model')
    logging.info('epoch: %d'%epoch)
    model.eval()
    loss, CPI_reconstruct = model()
    y_pred = CPI_reconstruct[(test_triples[:,0],test_triples[:,1])].cpu().numpy()
    np.save('./experiments/pred/fold%d.npy'%fold,y_pred)
    y_true = test_triples[:,2]
    result = evaluate(fold, epoch,y_pred, y_true)
    logging.info("loss: %f"%loss.item())
    return result

def evaluate(fold, epoch, y_pred, y_true):
    threshold = 0.5
    test_auc = roc_auc_score(y_true, y_pred)
    test_aupr = average_precision_score(y_true, y_pred)
    y_pred = [1 if prob >= threshold else 0 for prob in y_pred]
    test_f1 = f1_score(y_true, y_pred)
    test_precision = precision_score(y_true, y_pred)
    test_recall = recall_score(y_true, y_pred)
    test_accuracy = accuracy_score(y_true, y_pred)
    # if test_aupr>aupr_max:
    result={'fold':fold+1,
            'epoch':epoch,
            'test_auc':test_auc,
            "test_aupr":test_aupr,
            'test_f1':test_f1,
            'test_precision':test_precision,
            'test_recall':test_recall,
            'test_accuracy':test_accuracy,
            'test_auc':test_auc, 
            "test_aupr": test_aupr,
            'test_f1':test_f1, 
            'test_precision':test_precision,
            'test_recall':test_recall,
            'test_accuracy':test_accuracy
            }
    logging.info(result)
    for key,value in result.items():
        logging.info("%s: %s"%(key,value))
    return result