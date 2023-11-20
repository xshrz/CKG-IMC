# def trainmodel(model,args):
import torch.nn.functional as F
import torch
import torch.nn as nn
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

def train_sparse_autoencoder(model, dataloader, num_epochs, learning_rate,l1_penalty, weight_decay):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    for epoch in range(num_epochs):
        total_loss = 0.0
        # total_l1_loss = 0.0
        for data in dataloader:
            inputs = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            # l1 regularization to encourage sparsity
            # l1_loss = l1_penalty * torch.sum(torch.square(outputs))
            # l1_loss = l1_penalty * torch.sum(torch.abs(outputs))
            l1_loss = l1_penalty * torch.norm(model.encoder.weight,p=1)
            loss +=  l1_loss
            # l1_loss = l1_penalty * F.mse_loss(outputs,torch.zeros_like(outputs))

            loss.backward()
            optimizer.step()
            # total_l1_loss += l1_loss.item()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

    print("Training completed!")
def train_pna_imc_model_step(model,optimizer):
    model.train()
    optimizer.zero_grad()
    loss, CPI_reconstruct = model()
    loss.backward()
    optimizer.step()