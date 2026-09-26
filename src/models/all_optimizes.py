def optimize(self,g,ini_adj,x,labels,idx_train,end_epoch,epoch,teacher_output,teacher_rep):
        self.train()

    # ======================================================
    # Phase 1: Update GNN + classifier (feature learner)
    # ======================================================
    # Enable gradients only for GNN and classifier
        self.GNN.requires_grad_(True)
        self.classifier.requires_grad_(True)
        self.adv.requires_grad_(False)
        self.adj.requires_grad_(False)

        # Forward
        rep_base = self.GNN(self.g_t, self.x_t, self.weights)
        logits = self.classifier(rep_base)
        self.dist_loss = self.dist_criterion(F.log_softmax(logits / self.args.temperature, dim=1),
            F.softmax(teacher_output / self.args.temperature, dim=1),
        ) * (self.args.temperature * self.args.temperature)
        self.cls_loss = self.criterion(
            logits[idx_train],
            labels[idx_train]
        )
        #dist_sim = self.sim_kd(rep_base,teacher_rep)
        self.G_loss = self.cls_loss

        # Backward
        self.optimizer_G.zero_grad()
        self.G_loss.backward()
        self.optimizer_G.step()

        # Detach representation so adversary does NOT affect GNN
        rep_fixed = rep_base.detach()

        # ======================================================
        # Phase 2: Update adversary + adjacency predictor
        # ======================================================
        # Freeze GNN, enable gradients for adv + adj
        self.GNN.requires_grad_(False)
        self.classifier.requires_grad_(False)
        self.adv.requires_grad_(True)
        self.adj.requires_grad_(True)

        # ---- Generate new features ----
        f0 = self.adv(rep_fixed)                     # [N, new_features]

        # ---- Predict adjacency ----
        adj_logits = self.adj(rep_fixed) # [N, N]
        #mask_existing = (ini_adj > 0).float()
        #mask_new = 1 - mask_existing

        #adj = (1 - self.args.graph_skip_conn) * (ini_adj) + \
        #    self.args.graph_skip_conn * mask_existing * adj_logits \
        #    + self.args.alpha_new * mask_new * adj_logits
        adj = (1 - self.args.graph_skip_conn) * (ini_adj) \
            + self.args.graph_skip_conn * ((1 - self.args.graph_beta)* adj_logits + self.args.graph_beta * self.initial_sim)
        adj.fill_diagonal_(0)
        #adj = adj + torch.eye(adj.size(0), device=adj.device)

        # ---- Build graph (soft edges, no hard threshold) ----
        src, dst = torch.nonzero(adj > 0, as_tuple=True)
        weights = adj[src, dst]

        g_t = dgl.graph((src, dst), num_nodes=adj.size(0))
        if self.args.new_features:
            x_t = torch.cat(
                [x[:, :-self.args.new_features],
                f0.reshape(-1, self.args.new_features)],
                dim=1
            )
        else:
            x_t = self.x_t

        # Store persistent state (DETACHED)
        self.g_t = g_t
        self.x_t = x_t.detach()
        self.weights = weights.detach()

        if epoch < end_epoch:
            with torch.no_grad():
                rep_t = self.GNN(g_t, x_t, weights)
                logits_t = self.classifier(rep_t)

            # ---- Losses for adv + adj ----
            # Diversity loss
            #Z = F.normalize(f0.T, dim=1)
            #K = Z @ Z.T
            #K = K - torch.diag_embed(torch.diag(K))
            div_loss = self.args.alpha * self.new_feature_independence_loss(f0)

            self.dist_loss = self.dist_criterion(F.log_softmax(logits_t / self.args.temperature, dim=1),
                F.softmax(teacher_output / self.args.temperature, dim=1),
            ) * (self.args.temperature* self.args.temperature)
            #dist_sim = self.sim_kd(rep_t,teacher_rep)
            A_loss =  self.criterion(
                logits_t[idx_train],
                labels[idx_train]
            ) + div_loss + self.args.dist_alpha * (self.dist_loss) 
            #
            #+ self.args.gamma * dist_sim

            adj_loss = self.args.beta * self.lpp_trace_loss(
                rep_t.detach(),   # GNN is fixed
                adj_logits
            )
            #print(adj_loss,div_loss,self.dist_loss)
            total_loss = A_loss + adj_loss

            # Backward
            self.optimizer_A.zero_grad()
            self.optimizer_adj.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adv.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.adj.parameters(), 1.0)
            self.optimizer_A.step()
            self.optimizer_adj.step()
            #self.my_lr_scheduler.step(total_loss)
            #self.my_lr_scheduler2.step(total_loss)

        '''
        self.adv.requires_grad_(True)
        self.GNN.requires_grad_(True)
        self.adj.requires_grad_(True)
        
        #if epoch <= end_epoch:
        if epoch + 1:
            #f0 = torch.rand((len(self.x),self.args.new_features))
            rep = self.rep
            #if epoch <= end_epoch:
            for t in range(self.num_steps):
                #rep = rep.clone()
                rep = rep.detach()
                rep.requires_grad_(True)
                f0 = self.adv(rep)

                adj_logits = self.adj(rep)
                #adj_logits = F.softshrink(adj_logits,lambd = self.args.delta)
                #adj_logits = torch.clamp_min(adj_logits, 0.5)
                #adj_logits = adj_logits.triu(1)
                #adj_logits = adj_logits + adj_logits.T
                #adj_logits = (adj_logits >= 0.5).float()
                
                
                #edge_probs = torch.sigmoid(adj_logits)
                #adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.args.temperature, probs=edge_probs).rsample()
                
                #adj_logits = adj_logits.triu(1)
                #adj_logits = adj_logits + adj_logits.T
                #adj_sampled.fill_diagonal_(0)
                #adj_sampled = self.normalize_adj(adj_sampled)
                #adj_norm = adj_sampled.norm(p=1)
                #mask_existing = (ini_adj > 0).float()
                #mask_new = 1 - mask_existing

                #adj = (ini_adj) + self.args.graph_skip_conn * adj_logits
                #mask_existing = (ini_adj > 0).float()
                #mask_new = 1 - mask_existing
                adj = (1 - self.args.graph_skip_conn) * (ini_adj) + \
                        self.args.graph_skip_conn  * adj_logits
                #adj.fill_diagonal_(1)
                #adj = (self.args.graph_skip_conn * (ini_adj) + (1 - self.args.graph_skip_conn) * adj_logits)
                #print(adj)
                #adj = self.normalize_adj(adj)
                src, dst = (adj>0).nonzero(as_tuple=True)
                weights = F.relu(adj[src, dst])
                g_t = dgl.graph((src, dst))
                


                #layer_norm = nn.LayerNorm(f0.size(1))  # normalize feature dim
                #f0 = layer_norm(f0)
                #self.Z_detached = self.adv(self.rep.detach())
                #f0 = torch.where(torch.abs(f0) > self.threshold,f0,0)
                x_t = torch.cat([x[:, :-self.args.new_features].clone(),
                    f0.reshape(-1, self.args.new_features)], dim=1)

                rep = self.GNN(g_t,x_t,weights)
            y = self.classifier(rep)
            self.rep = rep
            self.x = x_t
            self.g = g_t
            self.weights = weights
        '''
        
            #self.x = x_t.detach()
        #else:
            #y = self.classifier(self.GNN(self.g,self.x, self.weights))
        
        
        
        
        #else:
        #    rep = self.GNN(g,self.x)
        
        #logits_detached = self.classifier(self.rep.detach())
        '''
        f0 = self.adv(self.rep.detach())
        x_t = torch.cat([self.x[:, :-self.args.new_features].clone(),
                f0.reshape(-1, self.args.new_features)], dim=1)
        self.rep = self.GNN(g,x_t)
        y = self.classifier(self.rep)
        '''
        
        
        
        
        #self.div_loss = self.args.alpha * torch.norm(self.Z_detached.T @ self.Z_detached, p="fro")
        #self.div_loss = ((F.normalize(f0.T, dim=1) @ F.normalize(f0.T, dim=1).T).fill_diagonal_(0).sum() / (f0.size(1)*(f0.size(1)-1)))
        '''
        self.G_loss = self.criterion(y[idx_train].float(),labels[idx_train]) + self.args.dist_alpha * self.dist_criterion(F.log_softmax(teacher_output[idx_train], dim=1),F.log_softmax(y[idx_train], dim=1))
        #self.A_loss = self.criterion(logits_detached[idx_train].float(),labels[idx_train]) + self.args.alpha * self.div_loss
          # out-of-place diag zero
        
        if epoch <= end_epoch:
            K = F.normalize(f0.T, dim=1) @ F.normalize(f0.T, dim=1).T
            K = K - torch.diag_embed(torch.diag(K)) 
            self.div_loss = self.args.alpha * torch.clamp((K.sum() / (f0.size(1) * (f0.size(1) - 1))),min = 0)
            self.adj_loss = self.args.beta * self.lpp_trace_loss(self.rep,adj_logits)
            self.G_loss = self.G_loss + self.div_loss + self.adj_loss
        self.optimizer_G.zero_grad()
        #self.optimizer_A.zero_grad()
        #self.optimizer_adj.zero_grad()
        self.G_loss.backward(retain_graph=True)
        self.optimizer_G.step()
        '''
        
        #self.adj_loss.backward()
        #self.optimizer_adj.step()
        #if epoch <= end_epoch:
        #    self.my_lr_scheduler2.step(self.G_loss)
        #    self.my_lr_scheduler.step(self.G_loss)
        #self.optimizer_A.step()
        
        #if epoch <= end_epoch:
        '''
        H_frozen = self.rep.detach()  # don't update GCN here
        Z_aux = self.adv(H_frozen)  # updates only MLP
        #self.div_loss = torch.norm(Z_aux.T @ Z_aux, p="fro")
        K = F.normalize(Z_aux.T, dim=1) @ F.normalize(Z_aux.T, dim=1).T
        K = K - torch.diag_embed(torch.diag(K))   # out-of-place diag zero
        self.div_loss = self.args.alpha * (K.sum() / (Z_aux.size(1) * (Z_aux.size(1) - 1))) 
        #+ self.args.beta * Z_aux.abs().sum() + self.args.gamma * Z_aux.pow(2).sum()
        
        self.optimizer_A.zero_grad()
        self.div_loss.backward()
        #self.optimizer_A.step()
        self.my_lr_scheduler2.step(self.div_loss)
        '''
        
        
        
        
        ## update gnn
        
        '''
        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()
        f0 = (self.adv(self.rep.detach()))
        #print(f0.shape)
        #f0 = torch.where(torch.abs(f0) > self.threshold,f0,0)
        self.x[:,-self.args.new_features:] = f0.reshape(-1,self.args.new_features)
        if epoch <= end_epoch:
            self.rep = self.GNN(g,self.x)
            y = self.classifier(self.rep)
            #print(torch.norm(f0.view(-1),1))
            self.A_loss = self.criterion_adv(y[idx_train].float(),labels[idx_train])+self.args.alpha*self.criterion_cons(self.x[idx_train],labels[idx_train])
            self.A_loss.backward(retain_graph=True)
            #self.optimizer_A.step()
            #torch.nn.utils.clip_grad_norm_(self.adv.parameters(), max_norm=1.0)
            self.my_lr_scheduler.step(self.A_loss)
        '''