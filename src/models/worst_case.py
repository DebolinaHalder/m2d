import torch.nn as nn
from models.GCN import GCN,GCN_Body
from models.GAT import GAT,GAT_body
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.int64)
        print(index.shape,target.shape)
        index.scatter_(1, target.data.view(-1,1), 1)
        
        index_float = index.type(torch.FloatTensor)
        print(index_float.shape)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index == 1, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)



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

class WorstGNN(nn.Module):

    def __init__(self, nfeat, args,weights,class_num_list,loss):
        super(WorstGNN,self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.GNN_sens = GCN(nfeat,args.hidden,1,dropout)
        self.GNN = get_model(nfeat,args)
        if loss == 'ldam':
            self.criterion_weighted = LDAMLoss(cls_num_list=class_num_list, max_m=0.5, s=30, weight=None)
            self.classifier = NormedLinear(nhid,2)
        else:
            self.criterion_weighted = nn.BCEWithLogitsLoss()
            self.classifier = nn.Linear(nhid,1)
        
        self.adv = nn.Linear(nhid,1)
        

        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)
        self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        self.optimizer_S = torch.optim.Adam(self.GNN_sens.parameters(), lr = args.lr2, weight_decay = args.weight_decay)

        self.args = args
        
        self.criterion = nn.BCEWithLogitsLoss()

        self.G_loss = 0
        self.A_loss = 0
        self.S_loss = 0

    def forward(self,g,x):
        s = self.GNN_sens(g,x)
        z = self.GNN(g,x)
        y = self.classifier(z)
        
        return y,s
    
    def optimize(self,g,x,labels,idx_train,sens,idx_sens_train):
        self.train()      

        #update E, G
        
        

        s = self.GNN_sens(g,x)
        h = self.GNN(g,x)
        y = self.classifier(h)



        s_g = self.adv(h)

        s_score = torch.sigmoid(s.detach())
        # s_score = (s_score > 0.5).float()
        s_score[idx_sens_train]=sens[idx_sens_train].unsqueeze(1).float()
        y_score = torch.sigmoid(y)
        self.cov =  torch.abs(torch.mean((s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))))
        self.adv_loss = self.criterion(s_g,s_score)




        ######## optimizing S###########
        self.optimizer_S.zero_grad()
        
        
        # s_score = (s_score > 0.5).float()
        
        #cov = torch.abs(torch.mean((s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))))
        self.S_loss = self.criterion(s[idx_sens_train], sens[idx_sens_train].unsqueeze(1).float()) + self.args.gama * self.adv_loss - self.args.delta * self.cov
       
        self.S_loss.backward(retain_graph=True)
        self.optimizer_S.step()






        
        self.optimizer_G.zero_grad()
        self.cls_loss = self.criterion_weighted(y[idx_train].float(),labels[idx_train].unsqueeze(1).float())
                       
        #print(self.adv_loss)
        
        self.G_loss = self.cls_loss  + self.args.alpha * self.cov - self.args.beta * self.adv_loss   
        
        self.G_loss.backward()
        self.optimizer_G.step()

        ## update Adv
        
        self.optimizer_A.zero_grad()
        s_g = self.adv(h.detach())
        self.A_loss = self.criterion(s_g,s_score)
        self.A_loss.backward()
        self.optimizer_A.step()

        #print("done till here")
        

