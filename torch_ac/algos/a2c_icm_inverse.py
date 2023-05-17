import torch
import torch.nn.functional as F

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
import numpy as np
 # Add parent directory to system path
# Import my_module from parent directory
import sys
sys.path.append("/home/rmapkay/rl-starter-files") 
#print('hello',sys.path)
from curiosity_models import MinigridForwardDynamicsNet, MinigridStateEmbeddingNet, MinigridInverseDynamicsNet

from torch_ac.algos import A2CAlgo

class A2CAlgoICM_inverse(A2CAlgo):
    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, ir_coeff=0.001, reshape_reward=None):
        
        
        #print('self.entropy_ir_coeff',self.entropy_ir_coeff)
        #print('num_frames_per_proc',num_frames_per_proc)
        #print('recurrence',recurrence)
        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda,
                 entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                 rmsprop_alpha, rmsprop_eps, preprocess_obss, reshape_reward)
        
        self.intrinsic_reward_coeff = ir_coeff
        self.prediction_scale=10
        self.forward_dynamics_loss_coef= 0.2
        shape = (self.num_frames_per_proc, self.num_procs)
        self.total_rewards= torch.zeros(*shape, device=self.device)
#initialize intrinsic rewards
        self.intrinsic_rewards=torch.zeros(*shape, device=self.device)
        #print('obs space',envs[0].observation_space)
        self.obss_to_embed = [None] * (shape[0])
        #if you put.cuda to your network then the weights will be on GPU
        self.embedding_network=  MinigridStateEmbeddingNet().cuda() 
        self.forward_dynamics_model= MinigridForwardDynamicsNet(envs[0].action_space).cuda()
        self.inverse_dynamics_model= MinigridInverseDynamicsNet(envs[0].action_space).cuda()

        self.forward_dynamics_optimizer = torch.optim.RMSprop(self.forward_dynamics_model.parameters(), lr, alpha=rmsprop_alpha, eps=rmsprop_eps)
        self.state_embedding_optimizer = torch.optim.RMSprop(self.embedding_network.parameters(), lr, alpha=rmsprop_alpha, eps=rmsprop_eps)
        self.inverse_dynamics_optimizer= torch.optim.RMSprop(self.inverse_dynamics_model.parameters(), lr, alpha=rmsprop_alpha, eps=rmsprop_eps)
  
    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        #print('ac model recurrence',self.recurrence)
        
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            #print('preprocessed_obs',preprocessed_obs.image.shape)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs)
            action = dist.sample()
            #('action',action)
            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            #print('the reward is', reward)
            done = tuple(a | b for a, b in zip(terminated, truncated))
            #print('hi done',done)
            # Update experiences values
            self.obss[i] = self.obs
            self.obs = obs
            if self.acmodel.recurrent:
                #print('yes')
                self.memories[i] = self.memory
                self.memory = memory
                #print('mem',memory)
            self.masks[i] = self.mask  
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            #print('action',action)
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
                ##print('self.rewards[i]',self.rewards[i])
            self.log_probs[i] = dist.log_prob(action)
            #add this for observations to embedd:
            if i==self.num_frames_per_proc-1: #exclude the first frame and incclude the next state after the last frame
                self.obss_to_embed[0:i-1]= self.obss[1:]
                self.obss_to_embed[i]=self.obs
                #print('test,self.obss_to_embed',self.obss_to_embed)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            #print(" self.log_episode_return", self.log_episode_return)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)
            #print(" self.log_episode_num_frames", self.log_episode_num_frames)

            for i, done_ in enumerate(done): #for any done episode in any process we append it to log_return
                if done_:
                    #print("i",i)
                    #print('done',done_)
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
        
                    #print(" self.log_return", self.log_return)
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask #to reset when the episode is done
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

            ##add this for observations to embed:
            

        
        
        # Add advantage and return to experiences

        # preprocessed_obs = self.preprocess_obss(self.obs, device=self.device) #ucomment after test
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)
        
        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        
        ##I added this
        exps.obs_to_embed=[self.obss_to_embed[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)] #obs_to_embed excludes the first frame, they include the 1st frame up to number of frames
        
        exps.obs = self.preprocess_obss(exps.obs, device=self.device) #uncomment after test
        exps.obs_to_embed = self.preprocess_obss(exps.obs_to_embed, device=self.device) #uncomment after test
        ##print('new exps.obs',exps.obs.image.shape)
        ##print('self.actions',self.actions)
        exps.action = self.actions.transpose(0, 1).reshape(-1)#.to('cuda:5')#uncomment after test
        #print('exps.obs[:-1]',exps.obs.image[:-1][:][:][:].shape)
       #i added this
        with torch.no_grad():
            state_emb = self.embedding_network(exps.obs)
        ##print('state emb', state_emb.shape)
        #next_state_emb = self.embedding_network(exps.obs[1:])
        with torch.no_grad():
            next_state_emb = self.embedding_network(exps.obs_to_embed)
        #print('next state emb', next_state_emb) 
        #print('exps.action[:-1]',exps.action[:-1])
        with torch.no_grad():
           
            pred_next_state_emb= self.forward_dynamics_model(state_emb,exps.action)

        #print('pred_next_state_emb',pred_next_state_emb)
        self.intrinsic_rewards= (torch.norm(pred_next_state_emb - next_state_emb, dim=1, p=2).pow(2))*0.5#.to('cuda:5')
        #print('khara',self.intrinsic_rewards.requires_grad)
        #print('self.intrinsic_rewards',self.intrinsic_rewards)
       # self.intrinsic_rewards=torch.tensor(self.intrinsic_rewards.reshape((-1, 1)), device=self.device)
        #zero_tensor = torch.tensor([[0.]], device=self.device)

        # Concatenate the original tensor and the zero tensor along the first dimension
        #self.intrinsic_rewards = torch.cat((self.intrinsic_rewards, zero_tensor), dim=0)
     
        ##('self.intrinsic_rewards',self.intrinsic_rewards)
        #exps.forward_dynamics_loss= self.intrinsic_rewards.mean()
        ##print('self.rewards',self.rewards)
        self.total_rewards = (self.rewards.transpose(0, 1).reshape(-1)) + self.intrinsic_reward_coeff * self.intrinsic_rewards
        ##print('self.total_rewards',self.total_rewards)
        #fixing what I messed up
        

        for i in reversed(range(self.num_frames_per_proc)):
            #print('i',i)
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            #print('next_mask',next_mask)
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0
            #print('next_advantages',next_advantage)

            delta = self.total_rewards[i] + self.discount * next_value * next_mask - self.values[i]
            #print('delta',delta.shape)
            #print('khara',self.discount * self.gae_lambda * next_advantage * next_mask)
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask
            #print("self.advantages[i]",self.advantages[i])
            #print("self.gae",self.gae_lambda)

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        #print("self.actions",self.actions)
       
       
        #print("self.actions",exps.action)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        ##
        exps.intrinsic_rewards=self.intrinsic_rewards

        # Preprocess experiences

      

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)
        #print("self.log_done_counter",self.log_done_counter)f
        #print("self.log_return",self.log_return)

        logs = {
            "return_per_episode": self.log_return[-keep:], #u keep the log of the last #processes episode returns
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }
        #print("self.log_return[-keep:]",self.log_return[-keep:])
        #print('logs are',logs)
        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        #print('self.log_return',self.log_return)
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs
    
    def update_parameters(self, exps):
        # Compute starting indexes
        #print('ba2ra')
        inds = self._get_starting_indexes()
        ##print("indexes",inds)

        # Initialize update values

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0
   

        # Initialize memory

        if self.acmodel.recurrent:
            memory = exps.memory[inds]
            #print("memory",memory)
            #print("indexes",inds)
            #print("experiences",exps.keys())
        for i in range(self.recurrence):
            # Create a sub-batch of experience
            ##print('rec',self.recurrence)
            ##print('exps',exps)
            sb = exps[inds + i]
            #print('inds+i',inds+i)
            #print("sub-batch",sb.obs)
            # Compute loss

            if self.acmodel.recurrent:
                dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
                #print('dist',dist)
                #print('memory',memory)
            else:
                dist, value = self.acmodel(sb.obs)
            #print("the value function is",value)
            entropy = dist.entropy().mean()
            #print("sb.advantage.shape",sb.advantage.shape)
            #print("policy loss before averaging",-dist.log_prob(sb.action) * sb.advantage)
            policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean() #averaging over all observations, and over all processes
            ##print('policy_loss',policy_loss)

            value_loss = (value - sb.returnn).pow(2).mean() #same (averaged over all observations and processes)

            #Added this part for feed forward training of forward dynamics 
            
            state_emb = self.embedding_network(sb.obs) 
            ##print('state emb', state_emb.shape)
        #next_state_emb = self.embedding_network(exps.obs[1:])
            next_state_emb = self.embedding_network(sb.obs_to_embed) 
           ##print('next state emb', next_state_emb.shape) 
        
            pred_next_state_emb= self.forward_dynamics_model(state_emb,sb.action)
            #print('halu',state_emb.requires_grad) 
            forward_dynamics_loss= ((torch.norm(pred_next_state_emb - next_state_emb, dim=1, p=2).pow(2)).mean())*0.5
            ##print('forward dynamics loss',forward_dynamics_loss)

            #inverse dynamics loss:
            pred_dist= self.inverse_dynamics_model(state_emb,next_state_emb)
            #print('pred_dist',pred_dist)
            #print('sb.action',sb.action)
            inverse_dynamics_loss= -(pred_dist.log_prob(sb.action)).mean()
            #print('invserse dynamics loss',inverse_dynamics_loss)
            #logits_predicted_actions=F.log_softmax(predicted_actions, dim=1)
            #print('logits',logits_predicted_actions)
            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss+  self.prediction_scale*(self.forward_dynamics_loss_coef *forward_dynamics_loss+ (1-self.forward_dynamics_loss_coef)*inverse_dynamics_loss)
            #print('loss',loss)
            # Update batch values

            update_entropy += entropy.item()
            update_value += value.mean().item() #just averaging the values of the observations collected in experience
            #print("update value",update_value)
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            update_loss += loss
           

        # Update update values

        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss /= self.recurrence

        # Update actor-critic

        self.optimizer.zero_grad()
        self.forward_dynamics_optimizer.zero_grad()
        self.state_embedding_optimizer.zero_grad()
        self.inverse_dynamics_optimizer.zero_grad()
        #torch.autograd.set_detect_anomaly(True)
        #update_loss = update_loss #I added this I had .detach here
        # with torch.no_grad():
        #     update_loss.backward(retain_graph=True)
        update_loss.backward()
        #update_loss2.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5

        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.forward_dynamics_model.parameters(),self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.inverse_dynamics_model.parameters(),self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.embedding_network.parameters(),self.max_grad_norm)

        self.optimizer.step()
        self.state_embedding_optimizer.step()
        self.forward_dynamics_optimizer.step()
        self.inverse_dynamics_optimizer.step()
        

        # Log some values

        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm
        }

        return logs
    
    def pass_models_parameters(self):
        return self.embedding_network,self.forward_dynamics_model,self.inverse_dynamics_model,self.state_embedding_optimizer,self.forward_dynamics_optimizer,self.inverse_dynamics_optimizer