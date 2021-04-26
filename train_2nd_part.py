"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import MultipleEnvironments
from src.model import PPO
from src.process_2nd_part import eval
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="custom_model")
    parser.add_argument("--saved_episode",type =int, default="0")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    envs = MultipleEnvironments(opt.world, opt.stage, opt.action_type, opt.num_processes)
    model_mast = PPO(envs.num_states, envs.num_actions)
    model_1 = PPO(envs.num_states, envs.num_actions)
    model_2 = PPO(envs.num_states, envs.num_actions)
    model_1.eval()
   
    
    if torch.cuda.is_available():
        try:
            model_1.load_state_dict(torch.load("{}/ppo_assistance_{}_{}".format(opt.saved_path, opt.world, opt.stage)))
            model_1.cuda()
            print('model-1 is loaded cuda version')
        except:
            print('failed to load model-1')
        try:
            model_2.load_state_dict(torch.load("{}/ppo_secndpt_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage,opt.saved_episode)))
            model_2.cuda()
            print('model-2 is loaded cuda version')
        except:
            print('failed to load model-2')
    else:
        try:
            model_1.load_state_dict(torch.load("{}/ppo_assistance_{}_{}".format(opt.saved_path, opt.world, opt.stage),
                                        map_location=lambda storage, loc: storage))
            print('model-1 is loaded non cuda version')
        except:
            print('Failed to load model-1')

        try:
            model_2.load_state_dict(torch.load("{}/ppo_scendpt_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, opt.saved_episode),
                                        map_location=lambda storage, loc: storage))
            print('model-2 is loaded non cuda version')
        except:
            print('Failed to load non cuda model-2')


    model_mast.load_state_dict(model_2.state_dict())
    if torch.cuda.is_available():
        model_mast.cuda()
    model_mast.share_memory()
    process = mp.Process(target=eval, args=(opt, model_mast, envs.num_states, envs.num_actions))
    process.start()
    optimizer = torch.optim.Adam(model_mast.parameters(), lr=opt.lr)
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()
    curr_episode = opt.saved_episode
    while True:
        curr_episode += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        print('##############  restarting the training loop  ###################')
        while True:
            while True:
                logits, value = model_1(curr_states)
                policy = F.softmax(logits, dim=1)
                action = torch.argmax(policy).item()
                action = torch.tensor(action)
                action = action.view(-1)
                if torch.cuda.is_available():
                    [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
                else:
                    [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]
                state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
                # print('position is',info[0]['x_pos'])
                if info[0]['x_pos'] > 1000:
                    # print('starting sample collection')
                    break
                else:
                    state = torch.from_numpy(np.concatenate(state,0))
                    curr_states = state
    
            state = torch.from_numpy(np.concatenate(state,0))
            curr_states = state

            for _ in range(opt.num_local_steps):
                states.append(curr_states)
                logits, value = model_mast(curr_states)
                values.append(value.squeeze())
                policy = F.softmax(logits, dim=1)
                old_m = Categorical(policy)
                action = old_m.sample()
                actions.append(action)
                old_log_policy = old_m.log_prob(action)
                old_log_policies.append(old_log_policy)
                if torch.cuda.is_available():
                    [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
                else:
                    [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]

                state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            
                state = torch.from_numpy(np.concatenate(state, 0))
                if torch.cuda.is_available():
                    state = state.cuda()
                    reward = torch.cuda.FloatTensor(reward)
                    done = torch.cuda.FloatTensor(done)
                else:
                    reward = torch.FloatTensor(reward)
                    done = torch.FloatTensor(done)
                rewards.append(reward)
                dones.append(done)
                curr_states = state
                if done:
                    # print('samples collected ',len(states))
                    break

            if len(states)>= opt.num_local_steps:
                # print('entring training loop. states list size is ', len(states))
                _, next_value, = model_mast(curr_states)
                next_value = next_value.squeeze()
                old_log_policies = torch.cat(old_log_policies).detach()
                actions = torch.cat(actions)
                values = torch.Tensor(values).detach()
                # values = torch.cat(values).detach()
                states = torch.cat(states)
                gae = 0
                R = []
                for value, reward, done in list(zip(values, rewards, dones))[::-1]:
                    gae = gae * opt.gamma * opt.tau
                    gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
                    next_value = value
                    R.append(gae + value)
                R = R[::-1]
                R = torch.cat(R).detach()
                advantages = R - values
                for i in range(opt.num_epochs):
                    indice = torch.randperm(opt.num_local_steps * opt.num_processes)
                    for j in range(opt.batch_size):
                        batch_indices = indice[
                                        int(j * (opt.num_local_steps * opt.num_processes / opt.batch_size)): int((j + 1) * (
                                                opt.num_local_steps * opt.num_processes / opt.batch_size))]
                        logits, value = model_mast(states[batch_indices])
                        new_policy = F.softmax(logits, dim=1)
                        new_m = Categorical(new_policy)
                        new_log_policy = new_m.log_prob(actions[batch_indices])
                        ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                        actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                        torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) *
                                                        advantages[
                                                            batch_indices]))
                        # critic_loss = torch.mean((R[batch_indices] - value) ** 2) / 2
                        critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                        entropy_loss = torch.mean(new_m.entropy())
                        total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
                        optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model_mast.parameters(), 0.5)
                        optimizer.step()
                print("Episode: {}. Total loss: {}".format(curr_episode, total_loss))

                try:
                    
                    if os.path.exists('{}/ppo_scendpt_{}_{}_{}'.format(opt.saved_path,opt.world, opt.stage,(curr_episode-1))):
                        # print('removing past saved data of episode',curr_episode)
                        os.remove('{}/ppo_scendpt_{}_{}_{}'.format(opt.saved_path,opt.world, opt.stage,(curr_episode-1)))
                except:
                    print('failed to remove past saved model')
                
                torch.save(model_mast.state_dict(),
                        "{}/ppo_scendpt_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage,curr_episode))
                break
            else:
                print('reseting training ')
        opt.saved_episode = curr_episode
        

if __name__ == "__main__":
    opt = get_args()
    train(opt)