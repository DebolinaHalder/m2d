import torch.nn as nn
from models.GCN import GCN,GCN_Body
from models.GAT import GAT,GAT_body
from models.MLPA import MLPA
import torch
import pyro
import dgl
import numpy as np
from torch import linalg as LA
from models.MLPX import MLPX
from utils import *

def get_model(nfeat, args):
    if args.model == "GCN":
        model = GCN_Body(nfeat,args.hidden,args.dropout)
    elif args.model == "GAT":
        heads =  ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT_body(args.num_layers,nfeat,args.hidden,heads,args.dropout,args.attn_drop,args.negative_slope,args.residual)
    else:
        print("Model not implement")
        return

    return model
def fair_metric(output,labels,sens,idx):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]==1

    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)

    pred_y = (output[idx].squeeze()>0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    if sum(idx_s0_y1) == 0 or sum(idx_s1_y1) == 0:
            equality = 1
    else:
        equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))

    return parity,equality
class Augmentation(nn.Module):

    def __init__(self, nfeat, args,temperature = 1):
        super(Augmentation,self).__init__()

        self.temperature = temperature
        self.GNN = get_model(nfeat,args)
        self.classifier = nn.Linear(args.hidden,1)
        
        self.xmlp = MLPX(in_feats = args.num_hidden, n_hidden = args.num_hidden, out_feats = 1, dropout = 0.1)
        G_params = (list(self.GNN.parameters()) + list(self.classifier.parameters()))
        self.optimizer_G = torch.optim.Adam(G_params, lr = args.lr)

        self.xmlp_params = list(self.xmlp.parameters())
        self.optimizer_xmlp = torch.optim.Adam(self.xmlp_params, lr = args.lr)
        
        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()
        self.G_loss = 0
        self.xmlp_loss = 0
        self.relu = nn.ReLU()

    def forward(self,g,x):
        z = self.GNN(g,x)
        y = self.classifier(z)
        
        
        return z,y,self.idx_sens_train
    
    
    

    
    def optimize(self,g,x,labels,sens,budget,idx_train):
        self.train()
        self.xmlp.requires_grad_(False)
        self.optimizer_G.zero_grad()

        
        h = self.GNN(g,x)
        y = self.classifier(h)
        
        mask_logits = self.xmlp(h)
        mask_probs = torch.sigmoid(mask_logits)
        mask = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=mask_probs).rsample()
        x_new = torch.ones(x.shape[0], 1) * mask
        self.idx_sens_train = torch.where(x_new > 0)[0]
        #self.idx_sens_train = torch.LongTensor(self.idx_sens_train)
        

        idx_all = torch.where(x_new >= 0)[0]
        #idx_all = torch.LongTensor(idx_all)


        self.G_loss = self.criterion(y[self.idx_sens_train],sens[self.idx_sens_train].unsqueeze(1).float())
        self.G_loss.backward(retain_graph=True)

        self.optimizer_G.step()
        
        self.xmlp.requires_grad_(True)
        
        self.optimizer_xmlp.zero_grad()
        h = self.GNN(g,x)
        y = self.classifier(h)
        
        budget_cost = self.relu(torch.tensor(len(self.idx_sens_train)-budget))
        preds = torch.flatten((y>0).type_as(sens))
        self.parity, _ = fair_metric(labels,labels,preds,idx_train)
        self.xmlp_loss = self.args.beta*self.parity + budget_cost.float() - self.args.alpha * self.criterion(y[self.idx_sens_train],sens[self.idx_sens_train].unsqueeze(1).float())
        self.xmlp_loss.backward()
        self.optimizer_xmlp.step()