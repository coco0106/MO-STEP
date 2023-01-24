from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class linear_cqn(torch.nn.Module):
  

    def __init__(self, state_size, action_size, reward_size):
        super(linear_cqn, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size

        # S x A -> (W -> R^n). =>. S x W -> (A -> R^n)
        self.affine1 = nn.Linear(state_size + reward_size,
                                 (state_size + reward_size) * 16)
        self.affine2 = nn.Linear((state_size + reward_size) * 16,
                                 (state_size + reward_size) * 32)
        self.affine3 = nn.Linear((state_size + reward_size) * 32,
                                 (state_size + reward_size) * 64)
        self.affine4 = nn.Linear((state_size + reward_size) * 64,
                                 (state_size + reward_size) * 32)
        self.affine5 = nn.Linear((state_size + reward_size) * 32,
                                 action_size * reward_size)

    def H(self, Q, w, s_num, w_num):
        # mask for reordering the batch
        mask = torch.cat(
            [torch.arange(i, s_num * w_num + i, s_num)
             for i in range(s_num)]).type(LongTensor)
        
        #Q(batch*w_num,num_nodes,action_size,reward_size)
        
        reQ = Q.reshape(-1,Q.shape[1], self.action_size * self.reward_size
                     )[mask].transpose(1,2).reshape(-1,Q.shape[1], self.reward_size)
        #reQ(batch*w_num*action_size,num_nodes,reward_size)

        reQ_ext = reQ.repeat(w_num, 1,1)
        #reQ_ext(batch*w_num*w_num*action_size,num_nodes,reward_size)
        #w(batch*w_num,num_nodes,reward_size)
        w_ext = w.repeat(1, 1,self.action_size * w_num)
        #w_ext(batch*w_num,num_nodes,action_size*w_num*reward_size)
       
        w_ext = w_ext.transpose(1,2).reshape(-1, self.reward_size,w_ext.shape[1]).transpose(1,2)
        #w_ext(batch*w_num*action_size*w_num,num_nodes,reward_size)
       
        # produce the inner products
        prod = torch.stack(list(map(torch.bmm,reQ_ext.unsqueeze(2), w_ext.unsqueeze(3).cuda())),0).squeeze()
       
        
        #prod(batch*w_num*action_size*w_num,num_nodes)
        # mask for take max over actions and weights
        prod = prod.view(-1, self.action_size * w_num,prod.shape[1])
         #prod(batch*w_num,action_size*w_num,num_nodes)
        inds = prod.max(1)[1]
        #(batch*w_num,num_nodes)
       
        mask = ByteTensor(prod.size()).zero_()
        mask.scatter_(1, inds.data.unsqueeze(1), 1)
        mask = mask.view(-1, mask.shape[2],1).repeat(1,1, self.reward_size)
        
        #mask(batch*w_num*action_size,num_nodes,reward_size)
        #reQ_ext(batch*w_num*w_num*action_size,num_nodes,reward_size)
        # get the HQ
        HQ = reQ_ext.masked_select(Variable(mask)).view(-1, self.reward_size)

        return HQ

    # def H_(self, Q, w, s_num, w_num):
    #     reQ = Q.view(-1, self.reward_size)

    #     # extend preference batch
    #     w_ext = w.unsqueeze(2).repeat(1, self.action_size, 1).view(-1, 2)

    #     # produce hte inner products
    #     prod = torch.bmm(reQ.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()

    #     # mask for take max over actions
    #     prod = prod.view(-1, self.action_size)
    #     inds = prod.max(1)[1]
    #     mask = ByteTensor(prod.size()).zero_()
    #     mask.scatter_(1, inds.data.unsqueeze(1), 1)
    #     mask = mask.view(-1, 1).repeat(1, self.reward_size)

    #     # get the HQ
    #     HQ = reQ.masked_select(Variable(mask)).view(-1, self.reward_size)

    #     return HQ

    def forward(self, state, preference, w_num=1):
        #state(batch,num_nodes,hidden_dim)
        #preference(batch*w_num,num_nodes,reward_size)
        
        s_num = int(preference.size(0) / w_num)
       
        x = torch.cat((state.cuda(), preference.cuda()), dim=2)
        
        x = x.view(x.size(0),x.size(1), -1)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        x = F.relu(self.affine4(x))
        q = self.affine5(x)
        
        #(batch*w_num,num_nodes,reward_size,action_size)
        
        q = q.view(-1,q.shape[1], self.action_size, self.reward_size)
        

        hq = self.H(q.detach(), preference, s_num, w_num)
     

        return hq, q
