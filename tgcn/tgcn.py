import numpy as np
from fastdtw import fastdtw
import torch
import math
from multiprocessing.dummy import Pool 
import torch.nn as nn
from tgcn.utils import calculate_laplacian_with_self_loop
import itertools
import datetime
import pandas as pd
import math
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr


R = rpy2.robjects.r
DTW = importr('dtw')

# from line_profiler import LineProfiler

class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self._adj=adj
        self.laplacian=calculate_laplacian_with_self_loop(self._adj.cuda())
        self.weights = nn.Parameter(
            torch.zeros(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.zeros(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state,dtw_all,multi):
        batch_size, num_nodes = inputs.shape

           
        # inputs=inputs.unsqueeze(0)
        # hidden_state=hidden_state.unsqueeze(0)
        # dtw_all=dtw_all.unsqueeze(0)
        
        #laplacian(num_nodes,num_nodes)
        if multi==True:
            laplacian=self.laplacian.cuda()
   
            dtw_all=(dtw_all+torch.eye(num_nodes).cuda()+laplacian)/2
        else:
            dtw_all=dtw_all
        # print(inputs.shape)
        dtw_all=dtw_all.reshape(1,num_nodes,num_nodes)

        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state.cuda()), dim=2)
        #(batch_size,num_nodes,num_gru_units + 1)
       
    
        a_times_concat=torch.matmul(dtw_all.to(torch.float32),concatenation.to(torch.float32))
        
        # #  (num_nodes, num_gru_units + 1, batch_size)
        # concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # # (num_nodes, (num_gru_units + 1) * batch_size)
        # concatenation = concatenation.reshape(
        #     (num_nodes, (self._num_gru_units + 1) * batch_size)
        # )
        # # (num_nodes, (num_gru_units + 1) * batch_size)
        # a_times_concat = laplacian @ concatenation
        # (num_nodes, num_gru_units + 1, batch_size)
        # a_times_concat = a_times_concat.reshape(
        #     (num_nodes, self._num_gru_units + 1, batch_size)
        # )
        # # (batch_size, num_nodes, num_gru_units + 1)
        # a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)

        # (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        # (batch_size * num_nodes, output_dim)
       
        outputs = a_times_concat.to(torch.float32) @ self.weights + self.biases
        # (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

 


class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int,scaler):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._scaler=scaler
        self._adj=adj
        
        self.dtw=[]
        self.p3=list(torch.ones(input_dim*input_dim,1))
        self.graph_conv1 = TGCNGraphConvolution(self._adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0)
        self.graph_conv2 = TGCNGraphConvolution(self._adj, self._hidden_dim, self._hidden_dim)
    def inverse_norm(self,x):
        #(batch,num_nodes,time)
        result_x=self._scaler.inverse_transform(x).permute(2,0,1)
        return result_x
   
    def forward(self, inputs, states,his,epoch,iter):
        #(num_nodes,batch_size,time)
        input_inver=self.inverse_norm(inputs[:,:,:])
       
       
        num_nodes=inputs.shape[2]
        batch_size=inputs.shape[0]
        times=inputs.shape[1]
        #torch(batch_size,time,num_nodes), torch(time,batch_size,num_nodes*hidden_dim)
        gcn_1=torch.zeros(inputs.shape[0], self._input_dim * self._hidden_dim*2).cuda()
        gcn_2=torch.zeros(inputs.shape[0], self._input_dim * self._hidden_dim).cuda()
       
        p1=list(input_inver.repeat(num_nodes,1,1))
        
        df=pd.DataFrame({0:p1,2:self.p3})
        
       
        if epoch>0 and type(self.dtw) ==list:
            self.dtw=torch.cat(self.dtw,0).reshape(-1,12)

        for t in range(times):
            g=True
            p2=list(input_inver[:,:,:t+1].repeat(1,num_nodes,1).reshape(-1,batch_size,t+1))
            df[1]=p2
            
           
            if t==his and type(self.dtw)==torch.Tensor:
                dtw_all=self.dtw[iter,t]
                self.p3=list(dtw_all.reshape(num_nodes*num_nodes))
            else:
                if True in (df[2]>0.4).values:
                    print(datetime.datetime.now())

                    df[2][df[2]>0.4]= df[df[2]>0.4].apply(lambda x: math.exp(-0.01*R.dtw(R.matrix(x[0].cpu().detach().numpy(),nrow=20,ncol=times).transpose(),R.matrix(x[1].cpu().detach().numpy(),nrow=20,ncol=t+1).transpose()).rx('distance')[0][0]),axis=1)
                    df[2][df[2]<0.4]=0
                    dtw_all=torch.tensor(list(df[2].values)).reshape(num_nodes,num_nodes).cuda()
                    dtw_all=torch.where(torch.eye(num_nodes).cuda()==1,torch.zeros_like(dtw_all),dtw_all).cuda()
                elif t==times-1:
                    dtw_all=torch.zeros(num_nodes,num_nodes).cuda()
                else:
                    g=False
                if t==his:
                    self.p3=df[2]
                    if epoch==0:
                        self.dtw.append(dtw_all) #(seq_len,iter)
                        
            
            if g ==True:
                #dtw_all(batch_size,num_nodes,num_nodes)
                if t==times-1:
                    multi=True
                else: 
                    multi=False
                #torch(batch_size,num_nodes,time)
            
            
                input=inputs[:,t,:]
                hidden_state=states[t].cuda()
            
                gcn_1=gcn_1+self.graph_conv1(input,hidden_state,dtw_all,multi)
                
                # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
                concatenation = torch.sigmoid(gcn_1)
                # r u(batch_size, num_nodes, num_gru_units)
                r, u = torch.chunk(concatenation, chunks=2, dim=1)
                # c (batch_size, num_nodes * num_gru_units)
        
            
                gcn_2=gcn_2+self.graph_conv2(input,r*hidden_state,dtw_all,multi)
            
                c = torch.tanh(gcn_2)
                # h (batch_size, num_nodes * num_gru_units)
                new_hidden_state = u * hidden_state + (1.0 - u) * c
      
        return new_hidden_state, new_hidden_state

 

   




class TGCN(nn.Module):
    def __init__(self, adj, hidden_dim,scaler):
        super(TGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self._adj=adj
        self._scaler=scaler
        
        self.tgcn_cell = TGCNCell(self._adj, self._input_dim, self._hidden_dim,scaler)
        self.linear=nn.Linear(self._hidden_dim, 1)
   



    def forward(self, inputs,epoch,iter):
        
        batch_size, seq_len, num_nodes = inputs.shape
        encoder_state = torch.zeros(batch_size, num_nodes * self._hidden_dim)

        encoder_states=[]
        predictions=[]
        q_states=[]

        for i in range(seq_len):
            print("his",i)
           
           
            
            encoder_inputs=inputs[:,:i+1,:] #torch(batch_size,i+1,num_nodes)
            encoder_states.append(encoder_state)#(i+1,batch_size,num_nodes*hidden_dim)

            output,encoder_state = self.tgcn_cell(encoder_inputs, encoder_states,i,epoch,iter)
          
           
            output=output.reshape((batch_size, num_nodes, self._hidden_dim))
          
            q_states.append(output)
            decoder_state=encoder_state
            decoder_states=encoder_states+[decoder_state]
            decoder_input=torch.zeros(batch_size,1, num_nodes).cuda()
            
            decoder_inputs=torch.cat((encoder_inputs,decoder_input),1)
           
        
            for j in range (seq_len-i):
                print("pred",j)
                
                pred,decoder_state = self.tgcn_cell(decoder_inputs, decoder_states,13,-1,-1)
                pred = pred.reshape((batch_size, num_nodes, self._hidden_dim))
                prediction=self.linear(pred)
                decoder_input=prediction.transpose(1,2)
                
                decoder_inputs=torch.cat((decoder_inputs,decoder_input),1)#torch(batch_size,i+j+2,num_nodes)
                decoder_states=decoder_states+[decoder_state]#torch(i+j+2,batch_size,num_nodes*hidden_dim)
           
            predictions.append(prediction[:,:,0])
        
        return self._scaler.inverse_transform(torch.stack(predictions,0)), torch.stack(q_states,0)
