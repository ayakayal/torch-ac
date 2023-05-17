import torch
import torch.nn.functional as F

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
import numpy as np


from torch_ac.algos import A2CAlgo


class A2CAlgoStateCount(A2CAlgo):
    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None,intrinsic_reward_coeff=0.0001,reshape_reward=None):
        

        #print('num_frames_per_proc',num_frames_per_proc)
        #print('recurrence',recurrence)
        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda,
                 entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                 rmsprop_alpha, rmsprop_eps, preprocess_obss, reshape_reward)
        self.intrinsic_reward_coeff = intrinsic_reward_coeff
        self.dict_procs=[]
        for p in range(self.num_procs):
            train_state_count=dict()
            self.dict_procs.append(train_state_count)
        shape = (self.num_frames_per_proc, self.num_procs)
        self.total_rewards= torch.zeros(*shape, device=self.device)

        #initialize intrinsic rewards
        self.intrinsic_rewards=torch.zeros(*shape, device=self.device)
    def pass_models_parameters(self):
        return self.dict_procs
    
    def collect_experiences(self):
        shape = (self.num_frames_per_proc, self.num_procs)
        self.total_rewards= torch.zeros(*shape, device=self.device)

        #initialize intrinsic rewards
        self.intrinsic_rewards=torch.zeros(*shape, device=self.device)
  
        for i in range(self.num_frames_per_proc):
            
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            #print('preprocessed_obs',preprocessed_obs)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs)
            action = dist.sample()

            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            ##print('hi obs',obs)
            ##print('the image observation is', obs[0]['image'])
            temp_irewards=[]
            for p in range(self.num_procs):
                
                obs_tuple=tuple( obs[p]['image'].reshape(-1).tolist())
                ##print('obs tuple',obs_tuple)
                if obs_tuple in self.dict_procs[p]:
                    self.dict_procs[p][obs_tuple]+= 1
                else:
                    self.dict_procs[p][obs_tuple]=1
                temp_irewards.append(self.intrinsic_reward_coeff /np.sqrt(self.dict_procs[p][obs_tuple])) 
                ##print('the dictionary after observing frame',i,'is ',self.dict_procs[p])
                ##print('dict sum', sum(self.dict_procs[p].values()))
                ##print('temp_irewards', temp_irewards)
            self.intrinsic_rewards[i]=torch.tensor(temp_irewards)
            ##print('self.intrinsic_rewards[i]',self.intrinsic_rewards[i])
            done = tuple(a | b for a, b in zip(terminated, truncated))
            ##print('done', done)
            
            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask  
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            ##print('action',self.actions[i])
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
                ##print('self.rewards[i]',self.rewards[i])
            #self.intrinsic_rewards[i]= torch.tensor(self.intrinsic_reward_coeff /np.sqrt(self.train_state_count[obs_tuple])) 
            #self.intrinsic_rewards[i]=0
            ##print('self.intrinsic_rewards in frame',i,'is ',self.intrinsic_rewards[i])
            self.total_rewards[i]= self.intrinsic_rewards[i] + self.rewards[i]
            #print('self.total_rewards',self.total_rewards[i])
            self.log_probs[i] = dist.log_prob(action)
            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            #print(" self.log_episode_return", self.log_episode_return)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)
            #print(" self.log_episode_num_frames", self.log_episode_num_frames)

            for i, done_ in enumerate(done):
                if done_:
                    #print("i",i)
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    #print(" self.log_episode_return", self.log_episode_return)
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask #to reset when the episode is done
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            #print('i',i)
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            #print('next_mask',next_mask)
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0
            #print('next_advantages',next_advantage)
            delta = self.total_rewards[i] + self.discount * next_value * next_mask - self.values[i]
            
          
            #print('delta',delta)
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask #it was delta instead of delta_intrinsic
            #print("self.advantages[i]",self.advantages[i])
            #print("self.gae",self.gae_lambda)

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        #print("self.actions",self.actions)
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        #print("self.actions",exps.action)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.reward_total=self.total_rewards.transpose(0,1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage #this include the intrinsic reward in the advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        #print('the experiences ',exps)
        #print('the observations in the exp',exps.obs.get('image').shape)
        # Log some values

        keep = max(self.log_done_counter, self.num_procs)
        #print("self.log_done_counter",self.log_done_counter)
        #print("self.log_return",self.log_return)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }
        #print("logs",logs)

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs
    