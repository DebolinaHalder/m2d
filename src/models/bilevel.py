import torch.nn as nn
from models.GCN import GCN,GCN_Body
from models.GAT import GAT,GAT_body
import torch
import numpy as np


def fair_metric(output,labels,sens):
    val_y = labels.cpu().numpy()
    idx_s0 = sens.cpu().numpy()==0
    idx_s1 = sens.cpu().numpy()==1

    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)

    pred_y = (output.squeeze()>0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))

    return parity,equality

def get_model(nfeat, args):
    if args.model == "GCN":
        model = GCN_Body(nfeat,args.num_hidden,args.dropout)
    elif args.model == "GAT":
        heads =  ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT_body(args.num_layers,nfeat,args.num_hidden,heads,args.dropout,args.attn_drop,args.negative_slope,args.residual)
    else:
        print("Model not implement")
        return

    return model

class BilevelGNN(nn.Module):

    def __init__(self, nfeat, args):
        super(BilevelGNN,self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.GNN = get_model(nfeat,args)
        self.classifier = nn.Linear(nhid,1)
        self.adv = nn.Linear(nhid,1)

        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)
        self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr = args.lr, weight_decay = args.weight_decay)

        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()

        self.G_loss = 0
        self.A_loss = 0

    def forward(self,g,x):
        z = self.GNN(g,x)
        y = self.classifier(z)
        return y
    
    def optimize(self,g,x,labels,idx_train,sens,idx_sens_train):
        self.train()

        ### update E, G
        self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()
        h = self.GNN(g,x)
        y = self.classifier(h)



        s_g = self.adv(h)

        s_score = torch.sigmoid(s_g.detach())
        s_pred = (s_score > 0.5).float()
        print(s_pred)

        s_pred[idx_sens_train]=sens[idx_sens_train].unsqueeze(1).float()
        y_score = torch.sigmoid(y)
        
        self.cov =  torch.abs(torch.mean((s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))))
        self.parity, self.eq = fair_metric(y,labels,torch.flatten(s_pred))

        self.cls_loss = self.criterion(y[idx_train],labels[idx_train].unsqueeze(1).float())
        #self.adv_loss = self.criterion(s_g,s_score)                
        
        self.G_loss = self.cls_loss  + self.args.alpha * self.cov + self.args.beta * self.eq
        self.G_loss.backward()
        self.optimizer_G.step()

        ## update Adv
        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()
        s_g = self.adv(h.detach())
        self.A_loss = self.criterion(s_g[idx_sens_train],sens[idx_sens_train].unsqueeze(1).float()) - self.args.beta*self.eq
        self.A_loss.backward()
        self.optimizer_A.step()

