#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import PNAConv

class PNA_IMC_Model_cat(nn.Module):
    def __init__(self, graph_data, CPI_train, CPI_train_mask, compound_embedding, protein_embedding, args):
        super(PNA_IMC_Model_cat, self).__init__()
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
    