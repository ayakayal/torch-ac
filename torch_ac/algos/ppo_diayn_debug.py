import torch
import torch.nn.functional as F

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
import numpy as np
from torch_ac.algos import PPOAlgo
from diayn_models_debug import DIAYNModule, Discriminator

class PPOAlgoDIAYNDEBUG(PPOAlgo):
    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,intrinsic_reward_coeff=0.0001, num_skills=4,disc_lr=0.0001
                 ,reshape_reward=None): 
      
        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda,
                 entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                 adam_eps, clip_eps, epochs, batch_size, preprocess_obss,
                 reshape_reward)
        self.intrinsic_reward_coeff = intrinsic_reward_coeff
        self.num_skills=num_skills
        self.p_z=np.full(num_skills, 1.0 / num_skills)
        #print('p_z',self.p_z)
        self.EPS = 1E-6
        shape = (self.num_frames_per_proc, self.num_procs)
        self.total_rewards= torch.zeros(*shape, device=self.device)
        self.intrinsic_rewards=torch.zeros(*shape, device=self.device)
        self.im_module= DIAYNModule(device=self.device,discriminator_network=Discriminator(n_skills=num_skills, n_hidden_filters=256), learning_rate=disc_lr, adam_eps=adam_eps)
        
        shape_z=(self.num_frames_per_proc, self.num_procs,self.num_skills) #1 hot encoding
        self.z=torch.zeros(*shape_z, device=self.device, dtype=torch.int)
        self.next_z=torch.zeros(*shape_z, device=self.device, dtype=torch.int) #next round skills
        for p in range(self.z.shape[1]):
            z_one_hot = torch.zeros(num_skills)
            z_one_hot[self.sample_z()] = 1
            
            self.z[0,p,:]=  z_one_hot
        self.log_episode_return_int = torch.zeros(self.num_procs, device=self.device)
        self.log_return_int =  [0] * self.num_procs
       
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
            for p, done_ in enumerate(done):
                #print('p',p)
                #print('done',done_) #for any done episode in any process we append it to log_return
                if not done_ and i<self.num_frames_per_proc-1:
                    self.z[i+1,p,:]=self.z[i,p,:]
                    #print('z when done=false',self.z)
                elif done_ and i<self.num_frames_per_proc-1:
                    z_one_hot = torch.zeros(self.num_skills)
                    z_one_hot[self.sample_z()] = 1
                    self.z[i+1,p,:]=z_one_hot
                    #print('frame done',i)
                    #print('process done',p)
                    #print('skill before',self.z[i,p,:])
                    #print('skill after',self.z[i+1,p,:])
                    #print('z when done=true',self.z)
                elif not done_ and i==self.num_frames_per_proc-1:
                    self.next_z[0,p,:]=self.z[i,p,:]
                    #print('next skill',torch.argmax(self.next_z,dim=2))
                elif done_ and i==self.num_frames_per_proc-1:
                    z_one_hot = torch.zeros(self.num_skills)
                    z_one_hot[self.sample_z()] = 1
                    self.next_z[0,p,:]=z_one_hot
                    #print('last frame done',i)
                    #print('process done',p)
                    #print('skill before',self.z[i,p,:])
                    #print('next skill',torch.argmax(self.next_z,dim=2))
            #print('self.next_z',torch.argmax(self.next_z,dim=2))
            self.obss[i] = self.obs
            #print('skills',torch.argmax(self.z[i],dim=1))
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
            input_next_obs = self.preprocess_obss(self.obs, device=self.device) #this is the next obs)
            #print('input next obs',input_next_obs)
            if i<self.num_frames_per_proc-1:
                one_hot=self.z[i+1]
            elif i==self.num_frames_per_proc-1:
                one_hot=self.next_z[0]
            #print('true next label',torch.argmax(one_hot,dim=1))
            with torch.no_grad():
                unnormalized_probs=self.im_module.q_discriminator(input_next_obs)
                #print('unnormalized_probs',unnormalized_probs)
                log_q=F.log_softmax(unnormalized_probs, dim=1)
                #print('log_q',log_q)
                z=torch.argmax(one_hot, dim=1)
                #print('this is z',z)
                log_q_z=torch.gather(log_q, 1,z.view(-1, 1))
                #print('log_q_z',log_q_z)
            #print('this is p(z)',self.p_z)
                log_p_z=torch.log(torch.gather(torch.tensor(self.p_z,device=self.device), 0,z)+self.EPS)
                #print('log_p_z',log_p_z)
                self.intrinsic_rewards[i]= torch.squeeze(log_q_z)-log_p_z
            #print('self.intrinsic_rewards[i]',self.intrinsic_rewards[i].requires_grad)
            self.total_rewards[i]=  self.rewards[i] + self.intrinsic_reward_coeff * self.intrinsic_rewards[i] 
            self.total_rewards[i]/= (1+self.intrinsic_reward_coeff)
            # print('input_next_obs.image',input_next_obs.image.shape)
            # for nob,nz in zip(input_next_obs.image, z):
            #     print('nob',nob.shape)
            #     print('z',z.shape)
            # print('end')
            # rewards_int = [self.im_module.compute_intrinsic_reward(next_obs=nob,z=nz,p_z=self.p_z,EPS=self.EPS) \
            #                             for nob,nz in zip(input_next_obs.image, z)]
            # self.intrinsic_rewards[i] == torch.tensor(rewards_int,device=self.device,dtype=torch.float)
            # self.total_rewards[i]=  self.rewards[i] + self.intrinsic_reward_coeff * self.intrinsic_rewards[i] 
            
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            #print(" self.log_episode_return", self.log_episode_return)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)
            self.log_episode_return_int += self.intrinsic_rewards[i].clone().detach()
            #print(" self.log_episode_num_frames", self.log_episode_num_frames)
            
            for i, done_ in enumerate(done): #for any done episode in any process we append it to log_return
                if done_:
                    #print("process",i,"done")
                    
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    #print(" self.log_return", self.log_return)
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())
                    self.log_return_int.append(self.log_episode_return_int[i].item())

            self.log_episode_return *= self.mask #to reset when the episode is done
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask
            self.log_episode_return_int *= self.mask
        
        
        exps = DictList()
        #print('self.z',self.z)
        exps.skills=self.z.transpose(0,1).reshape(self.num_procs*self.num_frames_per_proc,self.num_skills)
        #print('exps.skills',exps.skills)
        b=[self.z[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        
        #print('exps.obs',exps.obs)
        #print('exps.skills2',b)
        z_targets=torch.argmax(exps.skills, dim=1)
        #print('long skills',z_targets.long())
        #print('z targets',z_targets)
        discriminator_loss=self.im_module.update(obs=exps.obs,z_targets=z_targets,max_grad_norm=self.max_grad_norm)
       
        # Add advantage and return to experiences

        #print('self.z before',torch.argmax(self.z,dim=2))
        
        self.z=self.next_z.clone()
        #print('self.z after',torch.argmax(self.z,dim=2))
        #print('self.z finally',self.z)
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1),skill=self.z[0][:][:])
            else:
                _, next_value = self.acmodel(preprocessed_obs,skill=self.z[0][:][:])


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
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        #print("self.actions",exps.action)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        #add exps.skills
      
    

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)
        #print("self.log_done_counter",self.log_done_counter)f
        #print("self.log_return",self.log_return)

        logs = {
            "return_per_episode": self.log_return[-keep:], #u keep the log of the last #processes episode returns
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "discriminator_loss":discriminator_loss,
            "return_int_per_episode": self.log_return_int[-keep:]
        }
        #print("self.log_return[-keep:]",self.log_return[-keep:])

        self.log_done_counter = 0
        self.log_return_int = self.log_return_int[-self.num_procs:]
        self.log_return = self.log_return[-self.num_procs:]
        #print('self.log_return',self.log_return)
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs
    def update_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
    
            # print('epoch ',_)
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values
                # print('all indexes',self._get_batches_starting_indexes())
                # print(len(self._get_batches_starting_indexes()))
                # print('indexes',inds)
                # print('len',len(inds))
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(sb.obs, memory * sb.mask,sb.skills)
                    else:
                        dist, value = self.acmodel(sb.obs,sb.skills)

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": np.mean(log_entropies),
            "value": np.mean(log_values),
            "policy_loss": np.mean(log_policy_losses),
            "value_loss": np.mean(log_value_losses),
            "grad_norm": np.mean(log_grad_norms)
        }

        return logs
