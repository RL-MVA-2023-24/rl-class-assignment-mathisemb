from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        self.Q1 = RandomForestRegressor()
    
    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample() # gymnasium Discrete
        else:
            return greedy_action(self.Q1,observation,env.action_space.n)

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.Q1, file)

    def load(self):
        with open('random_forest_regressor_10000.pkl', 'rb') as file:
            self.Q1 = pickle.load(file)


def greedy_action(Q,s,nb_actions):
    Qsa = []
    for a in range(nb_actions):
        sa = np.append(s,a).reshape(1, -1)
        Qsa.append(Q.predict(sa))
    return np.argmax(Qsa)


def collect_samples(env, horizon, disable_tqdm=False, print_done_states=False):
    s, _ = env.reset()
    #dataset = []
    S = []
    A = []
    R = []
    S2 = []
    D = []
    for _ in tqdm(range(horizon), disable=disable_tqdm):
        a = env.action_space.sample()
        s2, r, done, trunc, _ = env.step(a)
        #dataset.append((s,a,r,s2,done,trunc))
        S.append(s)
        A.append(a)
        R.append(r)
        S2.append(s2)
        D.append(done)
        if done or trunc:
            s, _ = env.reset()
            if done and print_done_states:
                print("done!")
        else:
            s = s2
    S = np.array(S)
    A = np.array(A).reshape((-1,1))
    R = np.array(R)
    S2= np.array(S2)
    D = np.array(D)
    return S, A, R, S2, D

"""
def rf_fqi(S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
    nb_samples = S.shape[0]
    Qfunctions = []
    SA = np.append(S,A,axis=1)
    for iter in tqdm(range(iterations), disable=disable_tqdm):
        if iter==0:
            value=R.copy()
        else:
            Q2 = np.zeros((nb_samples,nb_actions))
            for a2 in range(nb_actions):
                A2 = a2*np.ones((S.shape[0],1))
                S2A2 = np.append(S2,A2,axis=1)
                Q2[:,a2] = Qfunctions[-1].predict(S2A2)
            max_Q2 = np.max(Q2,axis=1)
            value = R + gamma*(1-D)*max_Q2
        Q = RandomForestRegressor()
        Q.fit(SA,value)
        Qfunctions.append(Q)
    return Qfunctions
"""

"""
if __name__ == "__main__": # we train and save our agent model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    torch.manual_seed(seed)

    # Evaluation agent
    
    eval_agent = ProjectAgent()
    #eval_agent.load()

    train_var = False

    if train_var:
        eval_agent.load()
        nb_samples = 75000
        S,A,R,S2,D = collect_samples(env, int(nb_samples))

        SA = np.append(S,A,axis=1)
        value = R.copy()

        #Q1 = RandomForestRegressor()
        eval_agent.Q1.fit(SA,value);

        print("training MSE:", np.mean((value-eval_agent.Q1.predict(SA))**2))

        eval_agent.save(path="random_forest_regressor_" + str(int(nb_samples)) + ".pkl")
    else:
        eval_agent.load()


    #eval_agent.save(path="dqn_ep" + str(init_nb_episode+max_episode) + "_2hid" + str(hidden_dim) + ".pth")
    #eval_agent.network.eval()
    
    test = True
    if test:
        from evaluate import evaluate_HIV, evaluate_HIV_population
        score = evaluate_HIV(agent=eval_agent, nb_episode=1)
        print('score:', score)
        score_pop = evaluate_HIV_population(agent=eval_agent, nb_episode=15)
        print('score population:', score_pop)
"""
