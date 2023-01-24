from tgcn.utils import load_graph_data
from tgcn.utils import load_dataset
from tgcn.utils import reward
from tgcn.utils import mean_batch
from morl.linear import linear_cqn
from morl.meta import meta_agent
import yaml
from tgcn.tgcn import TGCN
import torch
import time
import torch.optim as optim




def train(config, tgcn, agent, dataloader):
    
    print("begin trainning....")
    log_file = open("{}{}.log".format(
        config['save']['log_path'], "e_{}".format(config['save']['env_name'])),'w')
    optimizer_tgcn = optim.Adam(tgcn.parameters(), lr=config['train']['lr'])
    for epoch in range(config['train']['epoch_num']):
        print("epoch",epoch)
        terminal = False
        loss_agent = 0
        tgcn.train()
        optimizer_tgcn.zero_grad()
        probe = torch.tensor([0.8, 0.2]).repeat(config['data']['num_nodes'],1)
        
        tot_reward = []
        tot_loss_tgcn = []
        tot_loss_morl = []

     
        time_epoch_begin=time.time()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            print("iter",iter)
           
         
            cnt = 0
            reals = torch.tensor(y).squeeze().cuda()
            # (batch_size, seq_len, num_nodes)
            inputs = torch.tensor(x).squeeze().cuda()
            predictions, states = tgcn(inputs,epoch,iter)
            
            # predictions=torch.ones(config['tgcn']['seq_len'],config['data']['batch_size'],config['data']['num_nodes'])
            # states=torch.ones(config['tgcn']['seq_len'],config['data']['batch_size'],config['data']['num_nodes'],config['tgcn']['hidden_dim'])
          
            # (seq_len, batch_size, num_nodes)  (seq_len, batch_size, num_nodes * hidden_dim)
            time_batch_begin=time.time()

            while not terminal:
                print("cnt",cnt)
                
                actions = []
                for t in range(config['tgcn']['seq_len']):
                    
                   
                    # state(1, num_nodes,hidden_dim)
                    action = torch.tensor(list(map(agent.act,states[t])))
                    
                    if t!=0:
                        action=torch.where(torch.sum(torch.cat(actions,0),0)==0,action,torch.zeros_like(action))
                    if t ==config['tgcn']['seq_len']-1:
                        action=torch.where(torch.sum(torch.cat(actions,0),0)==0,torch.ones_like(action),torch.zeros_like(action))

                    actions.append(action)
                    
                actions = torch.stack(actions, 0)
               
                
                # (seq_len, num_reward)
                
                rewards = reward(actions, predictions, reals)
                
                loss_agent += agent.learn(states, rewards*-1, actions, terminal)
                
                if cnt > 30:
                    terminal = True
                    agent.reset()

                cnt = cnt+1
            time_batch_end=time.time()

            loss_tgcn = torch.mean(torch.sum(rewards, 0)[0])
            tot_reward.append(torch.sum(torch.mul(probe.transpose(0,1).cuda(),torch.sum(torch.sum(rewards, 0),1).cuda())))
            tot_loss_tgcn.append(loss_tgcn)
            tot_loss_morl.append(loss_agent)
            
            loss_tgcn.requires_grad_(True).backward()
           
            optimizer_tgcn.step()
        time_epoch_end=time.time()
        
        log_file.write("Epoch:{},time_epoch:{},time_batch:{},Reaward:{},Loss_tgcn:{},Loss_morl:{}".format(
            epoch+1, time_epoch_end-time_epoch_begin,time_batch_end-time_batch_begin,mean_batch(tot_reward), mean_batch(tot_loss_tgcn), mean_batch(tot_loss_morl)))


        if (epoch+1) % 10 == 0:
            agent.save(config['save']['save_patyh'], "m_{}_e_{}".format(
                config['save']['q_name'], config['save']['env_name']))
            torch.save(tgcn, "{}{}.pkl".format(config['save']['save_path'], "m_{}_e_{}".format(
                config['save']['q_name'], config['save']['env_name'])))


if __name__ == '__main__':
   
    with open('data/metr-la.yaml') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    adj_mx = load_graph_data(config['data']['graph_pkl_filename'])
    dataloader = load_dataset(
        config['data']['dataset_dir'], config['data']['batch_size'])
    scaler = dataloader['scaler']
    

    if config['train']['serialize']:
        model = torch.load("{}{}.pkl".format(config['train']['save_path'], "m.{}_e.{}".format(
            config['save']['q_name'], config['save']['env_name'])))
        tgcn = torch.load("{}{}.pkl".format(config['train']['save_path'], "m.{}_e.{}".format(
            config['save']['tgcn_name'], config['save']['env_name'])))
    else:
        model = linear_cqn(config['morl']['state_size'], config['morl']
                           ['action_size'], config['morl']['reward_size']).cuda()
        tgcn = TGCN(torch.tensor(adj_mx), config['tgcn']['hidden_dim'], scaler).cuda()
    agent = meta_agent(model, config, is_train=True)
   
    train(config, tgcn, agent, dataloader)
    
