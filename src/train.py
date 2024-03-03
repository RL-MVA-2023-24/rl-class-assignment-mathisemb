from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self, network):
        self.network = network
    
    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample() # gymnasium Discrete
        else:
            return greedy_action(self.network, observation)

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self):
        self.network.load_state_dict(torch.load("dqn_agent_1hid32_ep700.pth", map_location='cpu'))
        self.network.eval()

class DQN_network(nn.Module):
    def __init__(self, nb_state, hidden_dim, nb_action, nb_hid_lay):
        super(DQN_network, self).__init__()
        self.input_layer = torch.nn.Linear(nb_state, hidden_dim)
        self.hidden_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        #self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(nb_hid_lay - 1)])
        self.output_layer = torch.nn.Linear(hidden_dim, nb_action)
        self.activation = torch.nn.ReLU()
        self.hidden_dim = hidden_dim
        """
        self.activation = torch.nn.ReLU()
        #self.activation = torch.nn.LeakyReLU()
        self.normalization = torch.nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.hidden_dim = hidden_dim
        self.nb_hid_lay = nb_hid_lay
        """
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))

        x = self.activation(self.hidden_layer(x))
        """
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            #x = self.normalization(x)
            #x = self.dropout(x)
        """

        x = self.output_layer(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

class dqn_agent:
    def __init__(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.lr = config['learning_rate']
        self.nb_grad_steps = config['nb_grad_steps']

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode, init_nb_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        max_cum_reward = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            #self.gradient_step()
            for _ in range(self.nb_grad_steps): 
                self.gradient_step()

            # next transition
            step += 1

            # max_episode_steps = 200
            if step >= env._max_episode_steps:
                step = 0
                done = True

            if done:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)

                """
                if episode_cum_reward > max_cum_reward:
                    max_cum_reward = episode_cum_reward
                    path_best_model = "dqn_ep" + str(init_nb_episode+episode) + "_lr" + str(config['learning_rate']) + "_" + str(self.model.nb_hid_lay) + "hid" + str(self.model.hidden_dim) + ".pth"
                    torch.save(self.model.state_dict(), path_best_model)
                """

                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return#, path_best_model

"""
if __name__ == "__main__": # we train and save our agent model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    torch.manual_seed(seed)

    # DQN config
    
    config = {'nb_actions': env.action_space.n,
            'learning_rate': 0.001,
            'gamma': 0.95,
            'buffer_size': 1000000,
            'epsilon_min': 0.01,
            'epsilon_max': 1.,
            'epsilon_decay_period': 1000,
            'epsilon_delay_decay': 20,
            'batch_size': 20}
    
    config = {'nb_actions': env.action_space.n,
            'learning_rate': 0.001,
            'gamma': 0.98,
            'buffer_size': 1000000,
            'epsilon_min': 0.01,
            'epsilon_max': 1.,
            'epsilon_decay_period': 5000,
            'epsilon_delay_decay': 100,
            'batch_size': 512,
            'nb_grad_steps': 1}
    
    # DQN
    nb_state = env.observation_space.shape[0]
    nb_action = env.action_space.n 
    hidden_dim = 32
    DQN = DQN_network(nb_state, hidden_dim, nb_action, nb_hid_lay=6)

    # DQN agent
    agent = dqn_agent(config, DQN)

    # Evaluation agent
    eval_agent = ProjectAgent(DQN)
    #eval_agent.load()
    eval_agent.network.load_state_dict(torch.load("dqn_agent_1hid32_ep700.pth", map_location='cpu'))
    max_episode = 200
    #episode_return = agent.train(env, max_episode=max_episode, init_nb_episode=0)
    init_nb_episode = 700
    #eval_agent.network.load_state_dict(torch.load(path_best_model, map_location='cpu'))
    #eval_agent.save(path="dqn_agent_1hid" + str(hidden_dim) + "_ep" + str(init_nb_episode+max_episode) + ".pth")
    eval_agent.network.eval()
    
    test = True
    if test:
        from evaluate import evaluate_HIV, evaluate_HIV_population
        score = evaluate_HIV(agent=eval_agent, nb_episode=1)
        score_pop = evaluate_HIV_population(agent=eval_agent, nb_episode=15)
        print('score:', score)
        print('score population:', score_pop)
"""
