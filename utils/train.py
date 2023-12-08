import torch
import torch.nn as nn
import torch.nn.functional as F

def train_kge_step(model, optimizer, train_iterator, args):
    '''
    A single train step. Apply back-propation and return the loss
    '''

    model.train()

    optimizer.zero_grad()

    positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

    
    positive_sample = positive_sample.to(args.device)
    negative_sample = negative_sample.to(args.device)
    subsampling_weight = subsampling_weight.to(args.device)


    negative_score = model((positive_sample, negative_sample), mode=mode)

    if args.negative_adversarial_sampling:
        #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                            * F.logsigmoid(-negative_score)).sum(dim = 1)
    else:
        negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

    positive_score = model(positive_sample)

    positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

    if args.uni_weight:
        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
    else:
        positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

    loss = (positive_sample_loss + negative_sample_loss)/2
    
    if args.regularization != 0.0:
        #Use L3 regularization for ComplEx and DistMult
        regularization = args.regularization * (
            model.entity_embedding.norm(p = 3)**3 + 
            model.relation_embedding.norm(p = 3).norm(p = 3)**3
        )
        loss = loss + regularization
        regularization_log = {'regularization': regularization.item()}
    else:
        regularization_log = {}
        
    loss.backward()

    optimizer.step()

    log = {
        **regularization_log,
        'positive_sample_loss': positive_sample_loss.item(),
        'negative_sample_loss': negative_sample_loss.item(),
        'loss': loss.item()
    }

    return log

def train_pna_imc_model_step(model,optimizer):
    '''
    A single train step. Apply back-propation and return the loss
    '''
    model.train()
    optimizer.zero_grad()
    loss, CPI_reconstruct = model()
    loss.backward()
    optimizer.step()
    return loss