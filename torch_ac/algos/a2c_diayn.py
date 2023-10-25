import torch
import torch.nn.functional as F

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
import numpy as np
from torch_ac.algos import A2CAlgo
from diayn_models import Discriminator
class A2CAlgoDIAYN(A2CAlgo):
    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=1,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None,intrinsic_reward_coeff=0.0001, num_skills=4,reshape_reward=None):
        

      
        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda,
                 entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                 rmsprop_alpha, rmsprop_eps, preprocess_obss, reshape_reward)
        self.intrinsic_reward_coeff = intrinsic_reward_coeff
        self.num_skills=num_skills
        shape = (self.num_frames_per_proc, self.num_procs)
        self.total_rewards= torch.zeros(*shape, device=self.device)
        #initialize intrinsic rewards
        self.intrinsic_rewards=torch.zeros(*shape, device=self.device)
        self.q_discriminator= Discriminator(self.num_skills, 256).cuda()
        self.q_discriminator_optimizer= torch.optim.RMSprop(self.q_discriminator.parameters(), lr, alpha=rmsprop_alpha, eps=rmsprop_eps)
        #F.log_softmax(, dim=1)
        
        self.p_z=np.full(num_skills, 1.0 / num_skills)

        shape_z=(self.num_frames_per_proc, self.num_procs,self.num_skills) #1 hot encoding
        self.z=torch.zeros(*shape_z, device=self.device, dtype=torch.int)
        self.next_z=torch.zeros(*shape_z, device=self.device, dtype=torch.int) #next round skills
        #each actor initially samples a skill z
        for p in range(self.z.shape[1]):
            z_one_hot = torch.zeros(num_skills)
            z_one_hot[self.sample_z()] = 1
            
            self.z[0,p,:]=  z_one_hot
        #print('sampled skill is', self.z,' first elt',self.z[0])
        """Samples z from p(z), using probabilities in self._p_z."""
        
        # self.z=torch.zeros(*shape, device=self.device, dtype=torch.int)
        # #each actor initially samples a skill z
        # for j in range(self.z.shape[1]):
        #     self.z[0,j]=  self.sample_z()
        #print('sampled skill is', self.z,' first elt',self.z[0])
    def sample_z(self):
        return np.random.choice(self.num_skills, p=self.p_z)
    
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
        
    
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1),skill=self.z[i])
                else:
                    dist, value = self.acmodel(preprocessed_obs,skill=self.z[i,:])
           
            entropy = dist.entropy().detach() #I added detach here
          
            action = dist.sample()
            #print('action',action)
            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            #print('the reward is', reward)
            done = tuple(a | b for a, b in zip(terminated, truncated))
            #print('hi done',done)
            # Update experiences values
            for p, done_ in enumerate(done): #for any done episode in any process we append it to log_return
                if not done_ and i<self.num_frames_per_proc-1:
                    self.z[i+1,p,:]=self.z[i,p,:]
                    print('z when done=false',self.z)
                elif done_ and i<self.num_frames_per_proc-1:
                    z_one_hot = torch.zeros(self.num_skills)
                    z_one_hot[self.sample_z()] = 1
                    self.z[i+1,p,:]=z_one_hot
                    print('z when done=true',self.z)
                elif not done_ and i==self.num_frames_per_proc-1:
                    self.next_z[0,p,:]=self.z[i,p,:]
                    print('next z when done=false',self.next_z)
                elif done_ and i==self.num_frames_per_proc-1:
                    z_one_hot = torch.zeros(self.num_skills)
                    z_one_hot[self.sample_z()] = 1
                    self.next_z[0,p,:]=z_one_hot
                    print('next z when done=true',self.next_z)
                # elif done==True and p==self.num_procs-1:
                #     z_one_hot = torch.zeros(self.num_skills)
                #     z_one_hot[self.sample_z()] = 1
                #     self.z[0,p,:]=z_one_hot
            print('z',self.z)
            self.obss[i] = self.obs
            self.obs = obs #this is the next state
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
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
            preprocessed_next_obs = self.preprocess_obss(self.obs, device=self.device)
            unnormalized_probs=self.q_discriminator(preprocessed_next_obs)
            print('unnormalized_probs',unnormalized_probs)
            log_q=F.log_softmax(unnormalized_probs, dim=1)
            # torch.gather
            print('log',log_q)
            if i<self.num_frames_per_proc-1:
                one_hot=self.z[i+1]
            z=torch.argmax(one_hot, dim=1)
            print('this is z',z)
            log_q_z=torch.gather(log_q, 1,z.view(-1, 1))
            print('log_q_z',log_q_z)
            print('this is p(z)',self.p_z)
            log_p_z=torch.log(torch.gather(torch.tensor(self.p_z,device=self.device), 0,z))
            print('log_p_z',log_p_z)
            print('self.total_rewards[i]',self.rewards[i])
            self.intrinsic_rewards[i]= torch.tensor((torch.squeeze(log_q_z)-log_p_z))
            print('self.intrinsic_rewards[i]',self.intrinsic_rewards[i])   
     
            self.total_rewards[i]= self.intrinsic_rewards[i] + self.rewards[i] 
            
            self.log_probs[i] = dist.log_prob(action)

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
        ## store your skills    
        exps = DictList()
        # exps.skills= [self.z[i][j]
        #             for j in range(self.num_procs)
        #             for i in range(self.num_frames_per_proc)]
        # print('exps.skills',exps.skills)
        exps.skills=self.z.transpose(0,1).reshape(self.num_procs*self.num_frames_per_proc,self.num_skills)
        print('exps.skills',exps.skills)
        # Add advantage and return to experiences

       
        self.z=self.next_z
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1),skill=self.next_z[0])
            else:
                _, next_value = self.acmodel(preprocessed_obs,skill=self.next_z[0])

        for i in reversed(range(self.num_frames_per_proc)):
            #print('i',i)
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            #print('next_mask',next_mask)
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0
            #print('next_advantages',next_advantage)

            delta = self.total_rewards[i] + self.discount * next_value * next_mask - self.values[i]
            #print('delta',delta)
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
        #add exps.skills
      
        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

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

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        #print('self.log_return',self.log_return)
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

        
        
        

    
