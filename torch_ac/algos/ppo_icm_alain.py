import torch
import torch.nn.functional as F

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
import numpy as np
from torch_ac.algos import PPOAlgo
from copy import deepcopy
from collections import deque
from statistics import mean
from icm_models_alain import ICMModule, EmbeddingNetwork_RIDE, InverseDynamicsNetwork_RIDE, ForwardDynamicsNetwork_RIDE
def default_preprocess_obss(obss, device=None):
    return torch.tensor(obss, device=device)

class PPOAlgoICMAlain(PPOAlgo):
    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256,singleton_env=False, preprocess_obss=None,intrinsic_reward_coeff=0.0001, 
                 reshape_reward=None): 
      
        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda,
                 entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                 adam_eps, clip_eps, epochs, batch_size,singleton_env, preprocess_obss,
                 reshape_reward)
        
        self.intrinsic_reward_coeff =intrinsic_reward_coeff
        self.num_actions=envs[0].action_space.n
        self.im_module = ICMModule(emb_network = EmbeddingNetwork_RIDE(),
                                           inv_network = InverseDynamicsNetwork_RIDE(num_actions=self.num_actions, device=self.device),
                                           forw_network = ForwardDynamicsNetwork_RIDE(num_actions=self.num_actions, device=self.device),
                                           device = device)
        shape = (self.num_frames_per_proc, self.num_procs)
        self.rewards_int = torch.zeros(*shape, device=self.device)
        self.rewards_total = torch.zeros(*shape, device=self.device)

        #add these for intrinsic rewards
        self.log_episode_return_int = torch.zeros(self.num_procs, device=self.device)
        self.log_return_int =  [0] * self.num_procs
        
    def collect_experiences(self):
        for i in range(self.num_frames_per_proc):
            self.total_frames+=self.num_procs
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs)
            action = dist.sample()
            #('action',action)
            if self.vizualise_video==True:
                self.frames.append(np.moveaxis(self.env.envs[0].get_frame(), 2, 0))
           
            obs, reward, terminated, truncated, agent_loc, _ = self.env.step(action.cpu().numpy())
            for agent_state in agent_loc:
                if agent_state in self.state_visitation_pos.keys():
                    self.state_visitation_pos[agent_state] += 1
                else:
                    self.state_visitation_pos[agent_state] = 1
            for r in reward:
                if r!=0 and self.found_reward==0:
                    self.saved_frame_first_reward=self.total_frames
                    self.found_reward=1
                    continue
                if r!=0 and self.found_reward==1:
                    self.saved_frame_second_reward=self.total_frames
                    self.found_reward=2
                    continue
                if r!=0 and self.found_reward==2:
                    self.saved_frame_third_reward=self.total_frames
                    self.found_reward=3
                    continue
            #print('the reward is', reward)
            done = tuple(a | b for a, b in zip(terminated, truncated))
            for p in range(self.num_procs):
                obs_tuple=tuple( obs[p]['image'].reshape(-1).tolist())
                #print('obs tuple',len(obs_tuple))
                if obs_tuple in self.train_state_count:
                    self.train_state_count[obs_tuple]+= 1
                else:
                    self.train_state_count[obs_tuple]=1
            # if done:
            #     print('done aya')
            #     print('the observation is: ',obs)
            #     print('same as initial obs', self.temp)
            
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
            input_current_obs = preprocessed_obs
            input_next_obs = self.preprocess_obss(self.obs, device=self.device) # contains next_observations

            # FOR COMPUTING INTRINSIC REWARD, THE REQUIRED SHAPE IS JUST A UNIT -- i.e image of [7,7,3]; action of [1] (it is calculated one by one)
            # FOR UPDATING COUNTS (done IN BATCH for efficiency), the shape requires to have the batch-- i.e image of [batch,7,7,3]; action of [batch,1]

           
            rewards_int = [self.im_module.compute_intrinsic_reward(obs=ob,next_obs=nobs,actions=act) \
                                        for ob,nobs,act in zip(input_current_obs.image, input_next_obs.image, action)]

            self.rewards_int[i] = rewards_int_torch = torch.tensor(rewards_int,device=self.device,dtype=torch.float)
           
            #print('self.rewards_int',self.rewards_int[i])
            temp_rewards_int=self.intrinsic_reward_coeff*self.rewards_int[i]
            if self.singleton_env != 'False':
                # print('yes singleton intrinsic rewards')
                
                for idx in range(len( temp_rewards_int)):
                    #print(self.intrinsic_rewards[i])
                    if agent_loc[idx] in self.ir_dict.keys():
                        #print(self.intrinsic_rewards[i][idx])
                        self.ir_dict[agent_loc[idx]] +=  temp_rewards_int[idx].item()
                    else:
                        self.ir_dict[agent_loc[idx]] =  temp_rewards_int[idx].item()
            #print('self.ir_dict',self.ir_dict)
            self.intrinsic_reward_per_frame=torch.mean(self.intrinsic_reward_coeff*self.rewards_int[i])
            #print('self.intrinsic_reward_per_frame',self.intrinsic_reward_per_frame)
            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_return_int += rewards_int_torch
            #print(" self.log_episode_return", self.log_episode_return)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)
            for i, done_ in enumerate(done): #for any done episode in any process we append it to log_return
                if done_:
                    #print("i",i)
                    #print('done',done_)
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_return_int.append(self.log_episode_return_int[i].item())
        
                    #print(" self.log_return", self.log_return)
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

                    

            self.log_episode_return *= self.mask #to reset when the episode is done
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask
            self.log_episode_return_int *= self.mask


        shape_im = (self.num_frames_per_proc,self.num_procs, 7,7,3) # preprocess batch observations (num_steps*num_instances, 7 x 7 x 3)
        input_obss = torch.zeros(*shape_im,device=self.device)
        input_nobss = torch.zeros(*shape_im,device=self.device)

        # generate next_states (same as self.obss + an additional next_state of al the penvs)
        nobss = deepcopy(self.obss)
        nobss = nobss[1:] # pop first element and move left
        nobss.append(self.obs) # add at the last position the next_states

        for num_frame,(mult_obs,mult_nobs) in enumerate(zip(self.obss,nobss)): # len(self.obss) ==> num_frames_per_proc == number_of_step

            for num_process,(obss,nobss) in enumerate(zip(mult_obs,mult_nobs)):
                o = torch.tensor(obss['image'], device=self.device)
                no = torch.tensor(nobss['image'], device=self.device)
                input_obss[num_frame,num_process].copy_(o)
                input_nobss[num_frame,num_process].copy_(no)

        # 1.2. reshape to have [num_frames*num_procs, 7, 7, 3]
        input_obss = input_obss.view(self.num_frames_per_proc*self.num_procs,7,7,3)
        input_nobss = input_nobss.view(self.num_frames_per_proc*self.num_procs,7,7,3)
        input_actions = self.actions.view(self.num_frames_per_proc*self.num_procs,-1)
        forward_dynamics_loss,inverse_dynamics_loss=self.im_module.update(obs=input_obss,next_obs=input_nobss,actions=input_actions)

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        self.rewards_total = self.rewards + self.intrinsic_reward_coeff*self.rewards_int
        self.rewards_total /= (1+self.intrinsic_reward_coeff)


        for i in reversed(range(self.num_frames_per_proc)):
            #print('i',i)
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            #print('next_mask',next_mask)
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0
            #print('next_advantages',next_advantage)

            delta = self.rewards_total[i] + self.discount * next_value * next_mask - self.values[i]
            #print('delta',delta.shape)
            #print('khara',self.discount * self.gae_lambda * next_advantage * next_mask)
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask
        
       
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
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)
        #print("self.log_done_counter",self.log_done_counter)f
        #print("self.log_return",self.log_return)
        self.number_of_visited_states= len(self.train_state_count)
        #size of the grid and the possible combinations of object index, color and status
        self.state_coverage= self.number_of_visited_states #percentage of state coverage
        #self.state_coverage_position=len(self.state_visitation_pos)
        non_zero_count=0
        for key, value in self.state_visitation_pos.items():
            if value != 0:
                non_zero_count += 1
        self.state_coverage_position= non_zero_count

        logs = {
            "return_per_episode": self.log_return[-keep:], #u keep the log of the last #processes episode returns
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "return_int_per_episode": self.log_return_int[-keep:],
            "forward_dynamics_loss": forward_dynamics_loss.item(),
            "inverse_dynamics_loss": inverse_dynamics_loss.item(),
            "state_coverage": self.state_coverage,
            "frame_first_reward": self.saved_frame_first_reward,
            "frame_second_reward": self.saved_frame_second_reward,
            "frame_third_reward": self.saved_frame_third_reward,
            "state_visitation_pos":self.state_visitation_pos,
            "state_coverage_position":self.state_coverage_position,
            "reward_int_per_frame":self.intrinsic_reward_per_frame,
            "ir_dict":self.ir_dict
            
        }
        #print("self.log_return[-keep:]",self.log_return[-keep:])
       
        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_return_int = self.log_return_int[-self.num_procs:]
        #print('self.log_return',self.log_return)
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs, self.frames