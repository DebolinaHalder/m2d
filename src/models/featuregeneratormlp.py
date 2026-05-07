import torch.nn as nn
from models.GCN import GCN,GCN_Body
from models.GAT import GAT,GAT_body
import torch
import torch.nn.functional as F
from models.MLPX import MLPX
from models.MLPA import MLPA
from models.edge_generator import SymmetricEdgePredictor
from models.presonalAttention import PersonaAttention
import difftopk
import pyro
import dgl
import numpy as np
from models.basemlp import Basemlp
torch.set_flush_denormal(True)





class topk_crossEntropy(nn.Module):
    def __init__(self, top_k=1):
        super(topk_crossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.top_k = top_k
    
    def forward(self, input, target):
        
        loss = self.criterion(input, target)
        if self.top_k == 1:
            return torch.mean(loss)
        else:
            #print(self.top_k * loss.shape[0])
            #print(loss.shape)
            valid_loss, idxs = torch.topk(loss, int(self.top_k * loss.shape[0]))    
            return torch.mean(valid_loss)

class Feature_generator(nn.Module):

    def __init__(self, nfeat,x,rep, args):
        super(Feature_generator,self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.x_t = x
        
        X_norm = F.normalize(x[:,:x.shape[1] - args.new_features], p=2, dim=1)
       
        #self.initial_sim = self.initial_sim/(self.initial_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        self.num_steps = args.num_steps
        self.mlp = Basemlp(nfeat,args)
        
        self.adv = nn.Linear(nhid,args.new_features)
        #self.adv = MLPX(nhid,args.hidden,args.new_features)
        self.rep = rep
        self.threshold = torch.zeros(1)
        self.dist_criterion = nn.KLDivLoss(reduction = 'batchmean')
        self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr = args.lr2, weight_decay = args.weight_decay)
        #self.optimizer_A.param_groups.append({'params': self.threshold })
        
        self.my_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer_A, mode='min', factor=0.1, patience=3)
        #self.my_lr_scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_A, gamma=args.decayRate,last_epoch=-1)
        self.args = args
        if args.num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else: 
            self.criterion = nn.CrossEntropyLoss()
        self.criterion_adv = topk_crossEntropy(args.k)
        #self.criterion_cons = SupervisedContrastiveLoss(temperature=args.temperature)
        #self.criterion_adv = difftopk.TopKCrossEntropyLoss( diffsort_method='odd_even', inverse_temperature=2, p_k=[.5, 0., 0., 0., .5], n=2, m=2, distribution='cauchy', art_lambda=None, device='cpu', top1_mode='sm')
        self.G_loss = 0
        self.div_loss = 0
        self.A_loss = 0
        self.dist_loss = 0
        self.cls_loss = 0
        #self.adj = MLPA(in_feats = args.hidden, dim_h = args.num_hidden, dim_z =nfeat)
        self.adj = SymmetricEdgePredictor(in_dim = args.num_hidden, hidden_dim = args.hidden)
        #self.adj = PersonaAttention(args.hidden,args.num_hidden)
        self.adj_params = list(self.adj.parameters())
        self.optimizer_adj = torch.optim.Adam(self.adj_params, lr = args.lr3,weight_decay=args.weight_decay)
        self.my_lr_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer_adj, mode='min', factor=0.1, patience=3)
        self.adj_loss = 0
        G_params = (list(self.mlp.G_params))
        
        self.optimizer_G = torch.optim.Adam(G_params,lr=args.lr, weight_decay = args.weight_decay)

    def forward(self,g,x):
        
        rep,y = self.mlp(self.x_t)
        
        #y = torch.sigmoid(y)
        return y
    
    def normalize_adj(self,adj,epsilon = 0):
        
        # normalize adj with A = D^{-1/2} @ A @ D^{-1/2}
        D_norm = torch.diag(torch.pow(adj.sum(1), -0.5))
        adj = D_norm @ adj @ D_norm
        return adj
    
    def new_feature_independence_loss(self,Z, target_var=1.0, eps=1e-8):
        """
        Z: (N, d_new) learned new features

        - d_new > 1: decorrelate new features
        - d_new = 1: avoid degenerate (collapsed) feature
        """
        '''
        Z = Z - Z.mean(dim=0, keepdim=True)
        N, d_new = Z.shape

        if d_new == 1:
            # variance regularization only
            var = Z.var(unbiased=False)
            return (var - target_var) ** 2

        # covariance matrix
        C = (Z.T @ Z) / (N + eps)

        # penalize off-diagonal entries
        off_diag = C - torch.diag(torch.diag(C))
        loss = (off_diag ** 2).sum()
        return loss
        '''

        if Z.shape[1] == 1:
            return 0
    
        K = F.normalize(Z.T, dim=1) @ F.normalize(Z.T, dim=1).T
        K = K - torch.diag_embed(torch.diag(K)) 
        #self.div_loss = self.args.alpha * (K.sum() / (f0.size(1) * (f0.size(1) - 1))) 
        div_loss = self.args.alpha * torch.clamp((K.sum() / (Z.size(1) * (Z.size(1) - 1))),min = 0)
        return div_loss
    
    def lpp_trace_loss(self,Z, A):
        '''
        Z = F.normalize(Z, dim=1)
        deg = A.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg + 1e-8))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt

        L = torch.eye(A.size(0), device=A.device) - A_norm

        return torch.trace(Z.T @ L @ Z) / Z.size(0)
        '''
        d = torch.clamp(A.sum(dim=1), min=1e-8)
        loss = -self.args.beta * torch.log(d + 1e-10).sum()/A.shape[0] + self.args.gamma * (A ** 2).sum() / (A.shape[0] ** 2)
        return loss

        
        
    
    def relation_matrix(self,h):
        h = F.normalize(h, dim=1)
        return h @ h.T
    
    def sim_kd(self,h_t,h_s):
        R_s = self.relation_matrix(h_s)
        R_t = self.relation_matrix(h_t)
        return torch.mean((R_s - R_t) ** 2)

    def get_div_loss(self,x):
        F_all = F.normalize(x)
        K = F_all.T @ F_all
        K.fill_diagonal_(0)

        # Only penalize correlations involving new features
        mask = torch.zeros_like(K)
        mask[-self.args.new_features:, :] = 1
        mask[:, -self.args.new_features:] = 1
        div_loss = self.args.alpha * ((K ** 2) * mask).mean()
        return div_loss

    def optimize(self,g,ini_adj,x,labels,idx_train,end_epoch,epoch,teacher_output,teacher_rep):
        self.train()
        ##################################################
        # Forward pass
        ##################################################

        rep,l = self.mlp(self.x_t)

        f0 = self.adv(rep.detach())
        
        #adj_logits = adj_logits / (adj_logits.sum(dim=1, keepdim=True) + 1e-8)

# --- Row-normalize initial similarity ---
        
        

        # ----- Build node features -----

        if self.args.new_features:
            x_t = torch.cat(
                [x[:, :-self.args.new_features],
                f0.reshape(-1, self.args.new_features)],
                dim=1
            )
        else:
            x_t = self.x_t

        ##################################################
        # Second GNN pass on learned graph
        ##################################################

        rep_t,logits = self.mlp(x_t)
        

        ##################################################
        # Losses
        ##################################################

        cls_loss = self.criterion(
            logits[idx_train],
            labels[idx_train]
        )

        self.dist_loss = self.dist_criterion(
            F.log_softmax(logits[idx_train] / self.args.temperature, dim=1),
            F.softmax(teacher_output[idx_train] / self.args.temperature, dim=1),
        ) * (self.args.temperature**2)

        #div_loss = self.args.alpha * self.new_feature_independence_loss(f0)
        div_loss = self.args.alpha * self.get_div_loss(x_t)

        
        self.G_loss = cls_loss
        total_loss = (1 - self.args.dist_alpha)* cls_loss + self.args.dist_alpha * self.dist_loss + div_loss

        ##################################################
        # Backward (all models)
        ##################################################

        self.optimizer_G.zero_grad()
        self.optimizer_A.zero_grad()

        total_loss.backward()

        self.optimizer_G.step()
        torch.nn.utils.clip_grad_norm_(self.adv.parameters(), 1.0)
        
        self.optimizer_A.step()
          

        ##################################################
        # Update persistent graph state (IMPORTANT)
        ##################################################

        
        self.x_t = x_t.detach()
        
        
        
        