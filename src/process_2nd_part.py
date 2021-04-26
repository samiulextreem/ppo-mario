"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

from os import stat
from numpy import intp
import torch
from src.env import create_train_env
from src.model import PPO
import torch.nn.functional as F
from collections import deque
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import os

def eval(opt, global_model, num_states, num_actions):
    torch.manual_seed(123)
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = create_train_env(opt.world, opt.stage, actions)
    local_model = PPO(num_states, num_actions)
    Is_model_2_loaded = False

    if torch.cuda.is_available():
        local_model.cuda()
    local_model.eval()
    state = torch.from_numpy(env.reset())
    if torch.cuda.is_available():
        state = state.cuda()
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    if done:
        if torch.cuda.is_available():
            local_model.load_state_dict(torch.load("{}/ppo_assistance_{}_{}".format(opt.saved_path, opt.world, opt.stage,opt.saved_episode)))
    
        if torch.cuda.is_available() is False:
            
            local_model.load_state_dict(torch.load("{}/ppo_assistance_{}_{}".format(opt.saved_path, opt.world, opt.stage),
                                        map_location=lambda storage, loc: storage))
    while True:
        curr_step += 1
        
        logits, value = local_model(state)
        
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)
        if info['x_pos'] > 1000 and Is_model_2_loaded == False :
            try:
                local_model.load_state_dict(global_model.state_dict())
                Is_model_2_loaded = True
                print('------ testing with model-----------')
            except:
                print('failed to load secondary training model')

        if info['x_pos'] < 1000 and Is_model_2_loaded == True:
            try:
                if torch.cuda.is_available():
                    local_model.load_state_dict(torch.load("{}/ppo_assistance_{}_{}".format(opt.saved_path, opt.world, opt.stage,opt.saved_episode)))
                if torch.cuda.is_available() is False:
                    local_model.load_state_dict(torch.load("{}/ppo_assistance_{}_{}".format(opt.saved_path, opt.world, opt.stage),
                                        map_location=lambda storage, loc: storage))
                Is_model_2_loaded = False
                print('assistance model loaded')
            except:
                print('failed to load secondary training model')

        # Uncomment following lines if you want to save model whenever level is completed
        if info['flag_get'] == True:
            print("###############  The model is finished .saving the model ###############")
            torch.save(local_model.state_dict(),
                        "{}/ppo_sendpt_finished_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage,opt.saved_episode))
            exit()
        
        havedisplay = "DISPLAY" in os.environ
        if havedisplay:
            env.render()
        actions.append(action)
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)
        if torch.cuda.is_available():
            state = state.cuda()
