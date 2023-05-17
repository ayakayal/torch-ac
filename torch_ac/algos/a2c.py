import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo


class A2CAlgo(BaseAlgo): #i changed it to BaseAlgoCount/BaseAlgoCountProcs it was BaseAlgo
    """The Advantage Actor-Critic algorithm."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None):
        print('change here')
        print('change here again')
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self, exps):
        # Compute starting indexes

        inds = self._get_starting_indexes()
        #print("indexes",inds)

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
            
            sb = exps[inds + i]
            #print('inds+i',inds+i)
            #print("sub-batch",sb.obs['image'])
            # Compute loss

            if self.acmodel.recurrent:
                dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
                #print('value',value)
                #print('memory',memory)
            else:
                dist, value = self.acmodel(sb.obs)
            #print("the value function is",value)
            entropy = dist.entropy().mean()
            #print("sb.advantage.shape",sb.advantage.shape)
            #print("policy loss before averaging",-dist.log_prob(sb.action) * sb.advantage)
            policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean() #averaging over all observations, and over all processes

            value_loss = (value - sb.returnn).pow(2).mean() #same (averaged over all observations and processes)

            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
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
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm
        }

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
