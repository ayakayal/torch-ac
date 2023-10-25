import multiprocessing
import gymnasium as gym
import numpy as np


multiprocessing.set_start_method("fork")

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            agent_loc = env.agent_pos
            if terminated or truncated:
                #print('yes restarted')
                #env.seed(np.random.randint(1,5)) 
                obs, _ = env.reset()
                agent_loc = env.agent_pos
            conn.send((obs, reward, terminated, truncated,agent_loc, info))
        elif cmd == "reset":
            #env.seed(np.random.randint(1,5))   
            obs, _ = env.reset()
            agent_loc = env.agent_pos
            conn.send(obs)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = multiprocessing.Pipe()
            self.locals.append(local)
            p = multiprocessing.Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()[0]] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, terminated, truncated, info = self.envs[0].step(actions[0])
        agent_loc = self.envs[0].agent_pos
        if terminated or truncated:
            #print('obs before reset',obs)
            #add this to test
            #self.envs[0].seed(0)
            obs, _ = self.envs[0].reset()
            agent_loc = self.envs[0].agent_pos
            #print('yup reset')
        results = zip(*[(obs, reward, terminated, truncated, agent_loc,info)] + [local.recv() for local in self.locals])
        return results
    
 


    def render(self):
        raise NotImplementedError

def singleton_worker(conn, env):
    """
    The worker class interacts with each environment individually and sends the result of the interaction
    """
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            # no agent position since this is the built-in OpenAI method
            obs, reward, terminated, truncated,  info = env.step(data)
            agent_loc = env.agent_pos
            if terminated or truncated:
                obs, _ = env.reset(seed = 10005)
                agent_loc = env.agent_pos
            conn.send((obs, reward, terminated, truncated,agent_loc, info))
        elif cmd == "reset":
            obs, _ = env.reset(seed = 10005)
            agent_loc = env.agent_pos
            conn.send(obs)
        else:
            raise NotImplementedError
        
class SingletonParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes.
    The ParallelEnv class manages the workers and controls the interactions of multiple workers at the same time
    """
    def __init__(self, envs,wrapper = None,beta=None):
        assert len(envs) >= 1, "No environment given."
        self.beta = beta
        if wrapper is not None:
            self.envs = [wrapper(env,self.beta) if wrapper is not None else env for env in envs]
        else:
            self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.locals = []
        for env in self.envs[1:]:
            local, remote = multiprocessing.Pipe()
            self.locals.append(local)
            p = multiprocessing.Process(target=singleton_worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()
    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset(seed=10005)[0]] + [local.recv() for local in self.locals]
        print(self.envs[0].agent_pos)
        return results
    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, terminated, truncated, info = self.envs[0].step(actions[0])
        agent_loc = self.envs[0].agent_pos
        if terminated or truncated:
            obs, _ = self.envs[0].reset(seed=10005)
            agent_loc = self.envs[0].agent_pos
        results = zip(*[(obs, reward, terminated, truncated, agent_loc, info)] + [local.recv() for local in self.locals])
        return results
    def render(self):
        raise NotImplementedError