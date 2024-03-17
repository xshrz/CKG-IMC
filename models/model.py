#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import PNAConv

# https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
# https://github.com/thunlp/OpenKE
class KGEModel(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, kg_model_name='ComplEx'):
        super(KGEModel, self).__init__()
        # self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        if kg_model_name == 'ComplEx':
            self.score_function = self.ComplEx
        elif kg_model_name == 'DistMult':
            self.score_function = self.DistMult
        elif kg_model_name == 'TransD':
            self.score_function = self.TransD
        elif kg_model_name == 'TransH':
            self.score_function = self.TransH
        elif kg_model_name == 'TransR':
            self.score_function = self.TransR
        elif kg_model_name == 'TransE':
            self.score_function = self.TransE
        elif kg_model_name == 'HolE':
            self.score_function = self.HolE
        elif kg_model_name == 'RotatE':
            self.score_function = self.RotatE
        else:
            supported_models = ['ComplEx', 'DistMult', 'TransD', 'TransH', 'TransR', 'TransE', 'HolE', 'RotatE']
            raise ValueError(f'Invalid kg_model_name: {kg_model_name}. Supported models are: {", ".join(supported_models)}')

            
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        if hidden_dim % 2:
            raise ValueError('ComplEx\'s hidden dim should be divided by 2')
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        name = self.score_function.__name__ 

        if self.score_function.__name__ == 'TransR':
            self.transfer_matrix = nn.Embedding(nrelation, self.entity_dim * self.relation_dim)
            nn.init.xavier_uniform_(self.transfer_matrix.weight.data)
        if self.score_function.__name__ == 'TransH':
            self.norm_vector = nn.Embedding(nrelation, self.relation_dim)
            nn.init.xavier_uniform_(self.norm_vector.weight.data)
        if self.score_function.__name__ == 'TransD':
            self.ent_transfer = nn.Embedding(self.nentity, self.entity_dim)
            self.rel_transfer = nn.Embedding(self.nrelation, self.relation_dim)
            nn.init.uniform_(
                tensor = self.ent_transfer.weight.data, 
                a= -self.embedding_range.item(), 
                b= self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor = self.rel_transfer.weight.data, 
                a= -self.embedding_range.item(), 
                b= self.embedding_range.item()
            )
        logging.info(f'{self.score_function.__name__}')
        logging.info('KGEModel Parameter Configuration:')
        for name, param in self.named_parameters():
            logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
            if self.score_function.__name__ == 'TransR':
                self.r_transfer = self.transfer_matrix(sample[:,1])
            if self.score_function.__name__ == 'TransH':
                self.relation_norm = self.norm_vector(sample[:,1]).unsqueeze(1)
            if self.score_function.__name__ == 'TransD':
                self.h_transfer = self.ent_transfer(sample[:,0]).unsqueeze(1)
                self.r_transfer = self.ent_transfer(sample[:,1]).unsqueeze(1)
                self.t_transfer = self.ent_transfer(sample[:,2]).view(batch_size, negative_sample_size, -1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
            if self.score_function.__name__ == 'TransR':
                self.r_transfer = self.transfer_matrix(tail_part[:,1])
            if self.score_function.__name__ == 'TransH':
                self.relation_norm = self.norm_vector(tail_part[:,1]).unsqueeze(1)
            if self.score_function.__name__ == 'TransD':
                self.h_transfer = self.ent_transfer(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
                self.r_transfer = self.ent_transfer(tail_part[:, 1]).unsqueeze(1)
                self.t_transfer = self.ent_transfer(tail_part[:, 2]).unsqueeze(1)
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            if self.score_function.__name__ == 'TransR':
                self.r_transfer = self.transfer_matrix(head_part[:,1])
            if self.score_function.__name__ == 'TransH':
                self.relation_norm = self.norm_vector(head_part[:,1]).unsqueeze(1)
            if self.score_function.__name__ == 'TransD':
                self.h_transfer = self.ent_transfer(head_part[:,0]).unsqueeze(1)
                self.r_transfer = self.ent_transfer(head_part[:,1]).unsqueeze(1)
                self.t_transfer = self.ent_transfer(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
        else:
            raise ValueError('mode %s not supported' % mode)
        
        score = self.score_function(head, relation, tail, mode)
        
        return score


    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score
    
    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        relation = torch.chunk(relation, 2, dim=2)[0]
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    # https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/TransE.py
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score
    
    # https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/TransD.py
    def TransD(self, head, relation, tail, mode):
        def _resize(tensor, axis, size):
            shape = tensor.size()
            osize = shape[axis]
            if osize == size:
                return tensor
            if (osize > size):
                return torch.narrow(tensor, axis, 0, size)
            paddings = []
            for i in range(len(shape)):
                if i == axis:
                    paddings = [0, size - osize] + paddings
                else:
                    paddings = [0, 0] + paddings
            print (paddings)
            return F.pad(tensor, paddings = paddings, mode = "constant", value = 0)
        def _transfer(e, e_transfer, r_transfer):
            if e.shape[0] != r_transfer.shape[0]:
                e = e.view(-1, r_transfer.shape[0], e.shape[-1])
                e_transfer = e_transfer.view(-1, r_transfer.shape[0], e_transfer.shape[-1])
                r_transfer = r_transfer.view(-1, r_transfer.shape[0], r_transfer.shape[-1])
                e = F.normalize(
                    _resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
                    p = 2, 
                    dim = -1
                )			
                return e.view(-1, e.shape[-1])
            else:
                return F.normalize(
                    _resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
                    p = 2, 
                    dim = -1
                )
        head = _transfer(head, self.h_transfer, self.r_transfer)
        tail = _transfer(tail, self.t_transfer, self.r_transfer)
        if mode == 'head_batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        return self.gamma.item() - torch.norm(score, p=1, dim=2)
            
    # https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/TransH.py
    def TransH(self, head, relation, tail, mode):
        def _transfer(e, norm):
            norm = F.normalize(norm, p = 2, dim = -1)
            return e - torch.sum(e * norm, -1, True) * norm
        head = _transfer(head, self.relation_norm)
        tail = _transfer(tail, self.relation_norm)
        if mode == 'head_batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        return self.gamma.item() - torch.norm(score, p=1, dim=2)
    
    # https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/TransR.py
    def TransR(self, head, relation, tail, mode):
        def _transfer(e, r_transfer):
            r_transfer = r_transfer.view(-1, self.entity_dim, self.relation_dim)
            return torch.matmul(e, r_transfer)

        head = _transfer(head, self.r_transfer)
        tail = _transfer(tail, self.r_transfer)
    
        if mode == 'head_batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        return self.gamma.item() - torch.norm(score, p=1, dim=2)
    



    # https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/HolE.py
    def HolE(self, head, relation, tail, mode):
        def _conj(tensor):
            zero_shape = (list)(tensor.shape)
            one_shape = (list)(tensor.shape)
            zero_shape[-1] = 1
            one_shape[-1] -= 1
            ze = torch.zeros(size = zero_shape, device = tensor.device)
            on = torch.ones(size = one_shape, device = tensor.device)
            matrix = torch.cat([ze, on], -1)
            matrix = 2 * matrix
            return tensor - matrix * tensor

        def _mul(real_1, imag_1, real_2, imag_2):
            real = real_1 * real_2 - imag_1 * imag_2
            imag = real_1 * imag_2 + imag_1 * real_2
            result = torch.complex(real, imag)
            return result

        def _ccorr(a, b):
            a = _conj(torch.fft.fft(a, dim=-1))
            b = torch.fft.fft(b, dim=-1)
            res = _mul(a.real, a.imag, b.real, b.imag)
            res = torch.fft.ifft(res, dim=-1).real
            return res

        score = _ccorr(head, tail) * relation
        score = torch.sum(score, 2)
        return score

# class SparseAutoencoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(SparseAutoencoder, self).__init__()
#         # Encoder
#         self.encoder = nn.Linear(input_dim, hidden_dim)
#         # Decoder
#         self.decoder = nn.Linear(hidden_dim, input_dim)

#     def forward(self, x):
#         encoded = self.encoder(x)  # Input shape: (1, batch_size, input_dim)
#         decoded = self.decoder(encoded)  # Output shape: (1, batch_size, output_dim)
#         return decoded  # Remove the first dimension
    


class PNA_IMC_Model(nn.Module):
    def __init__(self, graph_data, CPI_train, CPI_train_mask, compound_embedding, protein_embedding, args):
        super(PNA_IMC_Model, self).__init__()
        device = args.device

        self.n_compound = args.n_compound
        self.n_protein = args.n_protein

        pna_n_layer = args.pna_n_layer
        dim_hidden = args.pna_hidden_dim

        dim_kge_compound = compound_embedding.shape[1]
        dim_kge_protein = protein_embedding.shape[1]
        dim_edge = args.pna_edge_dim

        dim_pred = args.imc_k
        dim_compound_final = args.imc_compound_dim
        dim_protein_final = args.imc_protein_dim

        self.edge_index = graph_data.edge_index.to(device)
        self.edge_weight = graph_data.edge_weight.to(device)
        self.edge_type = graph_data.edge_type.to(device)
        deg = graph_data.deg.to(device)
        self.activation = nn.LeakyReLU()


        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.edge_emb = nn.Embedding(4, dim_edge)

        self.FC_kge_compound_project = nn.Linear(dim_kge_compound,dim_hidden)
        self.FC_kge_protein_project = nn.Linear(dim_kge_protein,dim_hidden)

        self.pnaConvList = nn.ModuleList()

        for _ in range(pna_n_layer):
            conv = PNAConv(dim_hidden, dim_hidden,act=self.activation,aggregators=aggregators,scalers=scalers,train_norm=True,deg=deg,edge_dim=dim_edge)
            self.pnaConvList.append(conv)

        self.compound_embedding = compound_embedding.detach().to(device)
        self.protein_embedding = protein_embedding.detach().to(device)
        self.CPI = torch.tensor(CPI_train).float().to(device)
        self.CPI_train_mask = torch.tensor(CPI_train_mask).to(device)

        self.FC_final_compound = nn.Linear(dim_hidden*(pna_n_layer+1), dim_compound_final)
        self.FC_final_protein = nn.Linear(dim_hidden*(pna_n_layer+1), dim_protein_final)

        self.imc_layer = IMC(dim_compound_final,dim_protein_final, dim_pred)

        for parameter in self.parameters():
            nn.init.xavier_uniform_(parameter.data) if len(parameter.data.shape) >= 2 else None


    def cal_CPI_reconstruct_loss(self,CPI_reconstruct):
        tmp = torch.mul(self.CPI_train_mask, (CPI_reconstruct - self.CPI))
        CPI_reconstruct_loss = torch.sum(torch.mul(tmp, tmp))
        return CPI_reconstruct_loss,CPI_reconstruct
    

    def forward(self):
        compound = self.FC_kge_compound_project(self.compound_embedding)

        protein = self.FC_kge_protein_project(self.protein_embedding)
        
        entity = torch.concat([compound,protein])
        entity_cat = entity.clone()
        edge_attr = self.edge_weight*self.edge_emb(self.edge_type)

        for conv in self.pnaConvList:
            entity = self.activation(conv(entity,self.edge_index, edge_attr))
            entity_cat = torch.cat([entity_cat,entity.clone()],dim=1)

        # entity = torch.concat([entity,entity_],dim=1)

        compound = entity_cat[:self.n_compound]
        protein = entity_cat[self.n_compound:self.n_compound+self.n_protein]

        final_compound = self.FC_final_compound(compound)
        final_compound = F.normalize(self.activation(final_compound)) 
        
        final_protein = self.FC_final_protein(protein) 
        final_protein = F.normalize(self.activation(final_protein))

        CPI_reconstruct = self.imc_layer(final_compound, final_protein)
        
        CPI_reconstruct_loss,CPI_reconstruct = self.cal_CPI_reconstruct_loss(CPI_reconstruct)

        return CPI_reconstruct_loss, CPI_reconstruct
    
class IMC(nn.Module):
    def __init__(self, in1_features, in2_features, dim_pred):
        super(IMC, self).__init__()
        self.W0p = nn.Parameter(torch.randn(in1_features, dim_pred))
        self.W1p = nn.Parameter(torch.randn(in2_features, dim_pred))
        self.params = nn.ParameterDict({
            'W0p': self.W0p,
            'W1p': self.W1p
        })

    def forward(self, x0, x1):
        return torch.matmul(torch.matmul(x0, self.W0p), torch.matmul(x1, self.W1p).transpose(0, 1))
        

    

class PNA_IMC_Model_without_kg(nn.Module):
    def __init__(self, graph_data, CPI_train, CPI_train_mask, compound_embedding, protein_embedding, args):
        super(PNA_IMC_Model_without_kg, self).__init__()
        device = args.device

        self.n_compound = args.n_compound
        self.n_protein = args.n_protein

        pna_n_layer = args.pna_n_layer
        dim_hidden = args.pna_hidden_dim
        # dim_kg = entity_kg_embedding.shape[1]
        dim_kge_compound = compound_embedding.shape[1]
        dim_kge_protein = protein_embedding.shape[1]
        dim_edge = args.pna_edge_dim

        dim_pred = args.imc_k
        dim_compound_final = args.imc_compound_dim
        dim_protein_final = args.imc_protein_dim

        self.edge_index = graph_data.edge_index.to(device)
        self.edge_weight = graph_data.edge_weight.to(device)
        self.edge_type = graph_data.edge_type.to(device)
        deg = graph_data.deg.to(device)

        self.compound_embedding = nn.Parameter(torch.randn((self.n_compound,dim_kge_compound)))
        self.protein_embedding = nn.Parameter(torch.randn((self.n_protein,dim_kge_protein)))

        self.activation = nn.LeakyReLU()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        self.edge_emb = nn.Embedding(4, dim_edge)

        self.FC_kge_compound_project = nn.Linear(dim_kge_compound,dim_hidden)
        self.FC_kge_protein_project = nn.Linear(dim_kge_protein,dim_hidden)

        self.pnaConvList = nn.ModuleList()
        for _ in range(pna_n_layer):
            conv = PNAConv(dim_hidden, dim_hidden,act=self.activation,
                           aggregators=aggregators,scalers=scalers,
                            train_norm=True,deg=deg,edge_dim=dim_edge)
            self.pnaConvList.append(conv)


        self.CPI = torch.tensor(CPI_train).float().to(device)
        self.CPI_train_mask = torch.tensor(CPI_train_mask).to(device)

        self.FC_final_compound = nn.Linear(dim_hidden+dim_hidden, dim_compound_final)
        self.FC_final_protein = nn.Linear(dim_hidden+dim_hidden, dim_protein_final)

        self.imc_layer = IMC(dim_compound_final,dim_protein_final, dim_pred)

        for parameter in self.parameters():
            nn.init.xavier_uniform_(parameter.data) if len(parameter.data.shape) >= 2 else None

        # logging.info('PNA_IMC_Model Parameter Configuration:')
        # for name, param in self.named_parameters():
        #     logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    def cal_CPI_reconstruct_loss(self,CPI_reconstruct):
        tmp = torch.mul(self.CPI_train_mask, (CPI_reconstruct - self.CPI))
        CPI_reconstruct_loss = torch.sum(torch.mul(tmp, tmp))
        return CPI_reconstruct_loss,CPI_reconstruct

    def forward(self):
        compound = self.FC_kge_compound_project(self.compound_embedding)
        protein = self.FC_kge_protein_project(self.protein_embedding)

        entity = torch.concat([compound,protein])
        entity_ = entity.clone()
        edge_attr = self.edge_weight*self.edge_emb(self.edge_type)
        for conv in self.pnaConvList:
            entity = conv(entity,self.edge_index, edge_attr)

        entity = torch.cat([entity,entity_],dim=1)

        compound = entity[:self.n_compound]
        protein = entity[self.n_compound:self.n_compound+self.n_protein]

        final_compound = self.FC_final_compound(compound)
        final_compound = F.normalize(self.activation(final_compound), dim=1) 
        
        final_protein = self.FC_final_protein(protein) 
        final_protein = F.normalize(self.activation(final_protein), dim=1)

        CPI_reconstruct = self.imc_layer(final_compound, final_protein)
        
        CPI_reconstruct_loss,CPI_reconstruct = self.cal_CPI_reconstruct_loss(CPI_reconstruct)

        return CPI_reconstruct_loss, CPI_reconstruct
    
class Inner_product_decoder(nn.Module):
    def __init__(self, in1_features, in2_features, dim_pred):
        super(Inner_product_decoder, self).__init__()
        self.MLP1 = nn.Sequential(
            nn.Linear(in1_features,in1_features//2),
            nn.LeakyReLU(),
            nn.Linear(in1_features//2,dim_pred),
            nn.LeakyReLU(),
            nn.Linear(dim_pred,dim_pred),
            nn.LeakyReLU(),
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(in2_features,in2_features//2),
            nn.LeakyReLU(),
            nn.Linear(in2_features//2,dim_pred),
            nn.LeakyReLU(),
            nn.Linear(dim_pred,dim_pred),
            nn.LeakyReLU(),
        )
        self.act = nn.LeakyReLU()
        self.FC1 = nn.Linear(in1_features,dim_pred)
        self.FC2 = nn.Linear(in2_features,dim_pred)
    def forward(self, x0, x1):
        # return torch.matmul(self.MLP1(x0),self.MLP2(x1).transpose(0,1))
        return torch.matmul(self.FC1(x0),self.FC2(x1).transpose(0,1))
        # return torch.matmul(self.act(self.FC1(x0)),self.act(self.FC2(x1).transpose(0,1)))

class MLP_decoder(nn.Module):
    def __init__(self, in1_features, in2_features, dim_pred,n_compound,n_protein):
        super(MLP_decoder, self).__init__()
        self.FC1 = nn.Linear(in1_features,dim_pred)
        self.FC2 = nn.Linear(in2_features,dim_pred)
        self.n_compound = n_compound
        self.n_protein = n_protein
        self.seq =nn.Sequential(
            nn.Linear((n_compound+n_protein)*dim_pred,20),
            nn.LeakyReLU(),
            nn.Linear(20,n_compound*n_protein),
            nn.LeakyReLU()
        )
    def forward(self, x0, x1):
        entity = torch.concat([self.FC1(x0),self.FC2(x1)]).reshape(-1)
        output = self.seq(entity).reshape(self.n_compound,self.n_protein)
        return output
from torch.utils.data import DataLoader
class MLP_decoder_right(nn.Module):
    def __init__(self, in1_features, in2_features, dim_pred,CPI_train_mask):
        super(MLP_decoder_right, self).__init__()
        self.CPI_train_mask = CPI_train_mask
        # self.compound_idx,self.protein_idx = torch.nonzero(CPI_train_mask,as_tuple=True)
        self.idx = torch.nonzero(CPI_train_mask)
        self.seq =nn.Sequential(
            nn.Linear(in1_features+ in2_features,dim_pred),
            nn.LeakyReLU(),
            nn.Linear(dim_pred,dim_pred//2),
            nn.LeakyReLU(),
            nn.Linear(dim_pred//2,1),
            nn.LeakyReLU()
        )
    def forward(self, x0, x1):
        n_compound,n_protein = x0.size(0),x1.size(0)
        dataloader = DataLoader(self.idx,batch_size=128,shuffle=False)
        output = torch.zeros((n_compound,n_protein)).to(x0.device)
        for idx in dataloader:
            data = torch.cat([x0[idx[:,0]],x1[idx[:,1]]],dim=1)
            output[idx[:,0],idx[:,1]] = self.seq(data).reshape(-1)
            # output = torch.cat([output,output_],dim=0)

        # x0 = x0.repeat((1,n_protein)).reshape(n_compound*n_protein,-1)
        # x1 = x1.repeat((n_compound,1))
        # entity = torch.cat([x0[self.compound_idx],x1[self.protein_idx]],dim=1)
        # dataloader = DataLoader(entity,batch_size=128,shuffle=False)
        # output = torch.tensor([])
        # for data in dataloader:
        #     output = torch.cat(output,self.seq(data))
        # entity = self.seq(entity).reshape(-1)
        # output_all = torch.zeros((n_compound,n_protein))
        # output_all[idx[:,0],idx[:,1]] = output
        # output = output.reshape(self.n_compound,self.n_protein)
        return output

class PNA_IMC_Model_without_IMC(nn.Module):
    def __init__(self, graph_data, CPI_train, CPI_train_mask, compound_embedding, protein_embedding, args):
        super(PNA_IMC_Model_without_IMC, self).__init__()
        device = args.device

        self.n_compound = args.n_compound
        self.n_protein = args.n_protein

        pna_n_layer = args.pna_n_layer
        dim_hidden = args.pna_hidden_dim

        dim_kge_compound = compound_embedding.shape[1]
        dim_kge_protein = protein_embedding.shape[1]
        dim_edge = args.pna_edge_dim

        dim_pred = args.imc_k
        dim_compound_final = args.imc_compound_dim
        dim_protein_final = args.imc_protein_dim

        self.edge_index = graph_data.edge_index.to(device)
        self.edge_weight = graph_data.edge_weight.to(device)
        self.edge_type = graph_data.edge_type.to(device)
        deg = graph_data.deg.to(device)
        self.activation = nn.LeakyReLU()


        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.edge_emb = nn.Embedding(4, dim_edge)

        self.FC_kge_compound_project = nn.Linear(dim_kge_compound,dim_hidden)
        self.FC_kge_protein_project = nn.Linear(dim_kge_protein,dim_hidden)

        self.pnaConvList = nn.ModuleList()

        for _ in range(pna_n_layer):
            conv = PNAConv(dim_hidden, dim_hidden,act=self.activation,aggregators=aggregators,scalers=scalers,train_norm=True,deg=deg,edge_dim=dim_edge)
            self.pnaConvList.append(conv)

        self.compound_embedding = compound_embedding.detach().to(device)
        self.protein_embedding = protein_embedding.detach().to(device)
        self.CPI = torch.tensor(CPI_train).float().to(device)
        self.CPI_train_mask = torch.tensor(CPI_train_mask).to(device)

        self.FC_final_compound = nn.Linear(dim_hidden+dim_hidden, dim_compound_final)
        self.FC_final_protein = nn.Linear(dim_hidden+dim_hidden, dim_protein_final)

        # self.imc_layer = MLP_decoder(dim_compound_final,dim_protein_final, dim_pred, args.n_compound, args.n_protein)
        # self.imc_layer = MLP_decoder_right(dim_compound_final,dim_protein_final, dim_pred,self.CPI_train_mask)
        self.imc_layer = Inner_product_decoder(dim_compound_final,dim_protein_final, dim_pred)

        for parameter in self.parameters():
            nn.init.xavier_uniform_(parameter.data) if len(parameter.data.shape) >= 2 else None


    def cal_CPI_reconstruct_loss(self,CPI_reconstruct):
        tmp = torch.mul(self.CPI_train_mask, (CPI_reconstruct - self.CPI))
        CPI_reconstruct_loss = torch.sum(torch.mul(tmp, tmp))
        return CPI_reconstruct_loss,CPI_reconstruct
    

    def forward(self):
        compound = self.FC_kge_compound_project(self.compound_embedding)

        protein = self.FC_kge_protein_project(self.protein_embedding)
        
        entity = torch.concat([compound,protein])
        entity_ = entity.clone()
        edge_attr = self.edge_weight*self.edge_emb(self.edge_type)

        for conv in self.pnaConvList:
            entity = conv(entity,self.edge_index, edge_attr)
            entity = self.activation(entity)

        entity = torch.concat([entity,entity_],dim=1)

        compound = entity[:self.n_compound]
        protein = entity[self.n_compound:self.n_compound+self.n_protein]

        final_compound = self.FC_final_compound(compound)
        final_compound = F.normalize(self.activation(final_compound)) 
        
        final_protein = self.FC_final_protein(protein) 
        final_protein = F.normalize(self.activation(final_protein))

        CPI_reconstruct = self.imc_layer(final_compound, final_protein)
        
        CPI_reconstruct_loss,CPI_reconstruct = self.cal_CPI_reconstruct_loss(CPI_reconstruct)

        return CPI_reconstruct_loss, CPI_reconstruct


class PNA_IMC_Model_without_pna(nn.Module):
    def __init__(self, graph_data, CPI_train, CPI_train_mask, compound_embedding, protein_embedding, args):
        super(PNA_IMC_Model_without_pna, self).__init__()
        device = args.device

        self.n_compound = args.n_compound
        self.n_protein = args.n_protein

        pna_n_layer = args.pna_n_layer
        dim_hidden = args.pna_hidden_dim

        dim_kge_compound = compound_embedding.shape[1]
        dim_kge_protein = protein_embedding.shape[1]
        dim_edge = args.pna_edge_dim

        dim_pred = args.imc_k
        dim_compound_final = args.imc_compound_dim
        dim_protein_final = args.imc_protein_dim

        self.edge_index = graph_data.edge_index.to(device)
        self.edge_weight = graph_data.edge_weight.to(device)
        self.edge_type = graph_data.edge_type.to(device)
        deg = graph_data.deg.to(device)
        self.activation = nn.LeakyReLU()


        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.edge_emb = nn.Embedding(4, dim_edge)

        self.FC_kge_compound_project = nn.Linear(dim_kge_compound,dim_hidden)
        self.FC_kge_protein_project = nn.Linear(dim_kge_protein,dim_hidden)

        self.pnaConvList = nn.ModuleList()

        for _ in range(pna_n_layer):
            conv = PNAConv(dim_hidden, dim_hidden,act=self.activation,aggregators=aggregators,scalers=scalers,train_norm=True,deg=deg,edge_dim=dim_edge)
            self.pnaConvList.append(conv)

        self.compound_embedding = compound_embedding.detach().to(device)
        self.protein_embedding = protein_embedding.detach().to(device)
        self.CPI = torch.tensor(CPI_train).float().to(device)
        self.CPI_train_mask = torch.tensor(CPI_train_mask).to(device)

        self.FC_final_compound = nn.Linear(dim_hidden+dim_hidden, dim_compound_final)
        self.FC_final_protein = nn.Linear(dim_hidden+dim_hidden, dim_protein_final)

        self.imc_layer = IMC(dim_compound_final,dim_protein_final, dim_pred)

        for parameter in self.parameters():
            nn.init.xavier_uniform_(parameter.data) if len(parameter.data.shape) >= 2 else None


    def cal_CPI_reconstruct_loss(self,CPI_reconstruct):
        tmp = torch.mul(self.CPI_train_mask, (CPI_reconstruct - self.CPI))
        CPI_reconstruct_loss = torch.sum(torch.mul(tmp, tmp))
        return CPI_reconstruct_loss,CPI_reconstruct
    

    def forward(self):
        compound = self.FC_kge_compound_project(self.compound_embedding)

        protein = self.FC_kge_protein_project(self.protein_embedding)
        
        entity = torch.concat([compound,protein])
        entity_ = entity.clone()
        # edge_attr = self.edge_weight*self.edge_emb(self.edge_type)

        # for conv in self.pnaConvList:
        #     entity = conv(entity,self.edge_index, edge_attr)
        #     entity = self.activation(entity)

        entity = torch.concat([entity,entity_],dim=1)

        compound = entity[:self.n_compound]
        protein = entity[self.n_compound:self.n_compound+self.n_protein]

        final_compound = self.FC_final_compound(compound)
        final_compound = F.normalize(self.activation(final_compound)) 
        
        final_protein = self.FC_final_protein(protein) 
        final_protein = F.normalize(self.activation(final_protein))

        CPI_reconstruct = self.imc_layer(final_compound, final_protein)
        
        CPI_reconstruct_loss,CPI_reconstruct = self.cal_CPI_reconstruct_loss(CPI_reconstruct)

        return CPI_reconstruct_loss, CPI_reconstruct
