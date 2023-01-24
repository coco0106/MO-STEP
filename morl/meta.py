from __future__ import absolute_import, division, print_function
import random
import torch
import copy
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from collections import deque

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class meta_agent(object):
    def __init__(self, model, config, is_train=False):
        self.model_ = model
        self.model = copy.deepcopy(model)
        self.is_train = is_train
        self.gamma = config['morl']['gamma']
        self.epsilon = config['train']['epsilon']
        self.epsilon_decay = config['train']['epsilon_decay']
        self.epsilon_delta = (
            config['train']['epsilon'] - 0.05) / config['train']['epoch_num']

        self.batch_size = config['data']['batch_size']
        self.weight_num = config['morl']['weight_num']

        self.beta = config['train']['beta']
        self.beta_init = config['train']['beta']
        self.homotopy = config['train']['homotopy']
        self.beta_uplim = 1.00
        self.tau = 1000.
        self.beta_expbase = float(
            np.power(self.tau*(self.beta_uplim-self.beta), 1./config['train']['epoch_num']))
        self.beta_delta = self.beta_expbase / self.tau

        self.trans_mem = deque()

        self.priority_mem = deque()
        self.optimizer = optim.Adam(
            self.model_.parameters(), lr=config['train']['lr'])

        self.w_kept = None
        self.update_count = 0
        self.update_freq = config['train']['update_freq']

        if self.is_train:
            self.model.train()
        if use_cuda:
            self.model.cuda()
            self.model_.cuda()

    def act(self, state, preference=None):
        # random pick a preference if it is not specified
        if preference is None:
            if self.w_kept is None:
                self.w_kept = torch.randn(state.shape[0],self.model_.reward_size)
                self.w_kept = (torch.abs(self.w_kept) /
                               torch.norm(self.w_kept, p=1))
            preference = self.w_kept
        
       
        _, Q = self.model_(Variable(state.unsqueeze(0)),
                           Variable(preference.unsqueeze(0)))
      
        Q = Q.view(-1,self.model_.action_size, self.model_.reward_size)
        
        Q = torch.bmm(Q.data, preference.unsqueeze(2).cuda()).squeeze()
        
        action = Q.max(1)[1].cpu().numpy()
        
        if self.is_train and torch.rand(1)[0] < self.epsilon:
            action = np.random.choice(self.model_.action_size, state.shape[0])
            
        return action

    def nontmlinds(self, terminal_batch):
        mask = ByteTensor(terminal_batch)
        inds = torch.arange(0, len(terminal_batch)).type(LongTensor)
        inds = inds[mask.eq(0)]
        return inds

    def learn(self, states, rewards, actions, terminal):
        if terminal:
            self.w_kept = None
            if self.epsilon_decay:
                self.epsilon -= self.epsilon_delta
            if self.homotopy:
                self.beta += self.beta_delta
                self.beta_delta = (self.beta-self.beta_init) * \
                    self.beta_expbase+self.beta_init-self.beta

        self.update_count += 1
        
        action_size = self.model_.action_size
        reward_size = self.model_.reward_size

        w_batch = np.random.randn(self.weight_num,states.shape[2], reward_size)
        w_batch = np.abs(w_batch) / \
            np.linalg.norm(w_batch, ord=2, axis=2, keepdims=True)
        w_batch = torch.from_numpy(w_batch.repeat(
            self.batch_size, axis=0)).type(FloatTensor)
        loss_all=[]
        
        for time in range(states.shape[0]):
            
            __, Q = self.model_(Variable(states[time].repeat(self.weight_num,1,1)),
                                Variable(w_batch), w_num=self.weight_num)
            if time != states.shape[0]-1:
                #(batch*w_num,num_nodes,reward_size)
                w_ext = w_batch.repeat(1, 1,action_size)
                #(batch*w_num,num_nodes,reward_size*action_size)
                w_ext = w_ext.transpose(1,2).reshape(-1, self.model.reward_size,w_ext.shape[1]).transpose(1,2)
                
                #(batch*w_num*action_size,num_nodes,reward_size)

                _, tmpQ = self.model_(Variable(states[time+1].repeat(self.weight_num,1,1), requires_grad=False),
                                    Variable(w_batch, requires_grad=False))
               
                tmpQ = tmpQ.transpose(1,2).reshape(-1,tmpQ.shape[1], reward_size)
               
                
                act = torch.stack(list(map(torch.bmm,
                                tmpQ.unsqueeze(2),Variable(w_ext.unsqueeze(3), requires_grad=False)))).squeeze().view(-1, action_size,states.shape[2]).max(1)[1]
                
                _, DQ = self.model(Variable(states[time+1].repeat(self.weight_num,1,1), requires_grad=False),
                                Variable(w_batch, requires_grad=False))
           
                HQ = DQ.gather(
                    2, act.view(-1, DQ.size(1),1, 1).expand(DQ.size(0),DQ.size(1), 1, DQ.size(3))).squeeze()
             
                with torch.no_grad():
                    Tau_Q = Variable(torch.zeros(
                        self.batch_size * self.weight_num,states.shape[2], reward_size).type(FloatTensor))
                    Tau_Q = self.gamma * HQ
                    
                    Tau_Q += Variable(rewards[time].permute(1,2,0).repeat(self.weight_num,1,1)).cuda()
            else:
                Tau_Q = Variable(rewards[time].permute(1,2,0).repeat(self.weight_num,1,1)).cuda().to(torch.float32)
            action = Variable(actions[time]).cuda()
            
            
            Q = Q.gather(2, action.repeat(self.weight_num,1).view(-1, Q.size(1),1, 1).expand(Q.size(0),Q.size(1), 1, Q.size(3))).squeeze()
         
            

            wQ = torch.stack(list(map(torch.bmm,Variable(w_batch.unsqueeze(2)),
                       Q.unsqueeze(3)))).squeeze()
           
           
            wTQ = torch.stack(list(map(torch.bmm,Variable(w_batch.unsqueeze(2)),
                       Tau_Q.to(torch.float32).unsqueeze(3)))).squeeze()

            # loss = F.mse_loss(Q.view(-1), Tau_Q.view(-1))
        
            loss = self.beta * F.mse_loss(wQ.view(-1), wTQ.view(-1))
            loss += (1-self.beta) * F.mse_loss(Q.view(-1), Tau_Q.view(-1))
            
           

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model_.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            loss_all.append(loss.data)

        if self.update_count % self.update_freq == 0:
            self.model.load_state_dict(self.model_.state_dict())

        return sum(loss_all)/len(loss_all)

    def reset(self):
        self.w_kept = None
        if self.epsilon_decay:
            self.epsilon -= self.epsilon_delta
        if self.homotopy:
            self.beta += self.beta_delta
            self.beta_delta = (self.beta-self.beta_init) * \
                self.beta_expbase+self.beta_init-self.beta

    # def predict(self, probe):
    #     return self.model(Variable(FloatTensor([0, 0]).unsqueeze(0), requires_grad=False),
    #                       Variable(probe.unsqueeze(0), requires_grad=False))

    def save(self, save_path, model_name):
        torch.save(self.model, "{}{}.pkl".format(save_path, model_name))

    def find_preference(
            self,
            w_batch,
            target_batch,
            pref_param):

        with torch.no_grad():
            w_batch = FloatTensor(w_batch)
            target_batch = FloatTensor(target_batch)

        # compute loss
        pref_param = FloatTensor(pref_param)
        pref_param.requires_grad = True
        sigmas = FloatTensor([0.001]*len(pref_param))
        dist = torch.distributions.normal.Normal(pref_param, sigmas)
        pref_loss = dist.log_prob(w_batch).sum(dim=1) * target_batch

        self.optimizer.zero_grad()
        # Total loss
        loss = pref_loss.mean()
        loss.backward()

        eta = 1e-3
        pref_param = pref_param + eta * pref_param.grad
        pref_param = simplex_proj(pref_param.detach().cpu().numpy())
        # print("update prefreence parameters to", pref_param)

        return pref_param


# projection to simplex
def simplex_proj(x):
    y = -np.sort(-x)
    sum = 0
    ind = []
    for j in range(len(x)):
        sum = sum + y[j]
        if y[j] + (1 - sum) / (j + 1) > 0:
            ind.append(j)
        else:
            ind.append(0)
    rho = np.argmax(ind)
    delta = (1 - (y[:rho+1]).sum())/(rho+1)
    return np.clip(x + delta, 0, 1)
