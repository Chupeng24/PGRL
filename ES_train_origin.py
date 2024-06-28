#!/usr/bin/env python3
import time
import numpy as np

from uniform_instance import FJSPDataset
from FJSP_Env import FJSP
from mb_agg import *
from copy import deepcopy
from Params import configs
from temp.validation_optimize_2_obj_3 import validate2

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


from policy import Policy
import random
import os
device = torch.device(configs.device)
MAX_BATCH_EPISODES = 100
MAX_BATCH_STEPS = 1200
NOISE_STD = 0.01
LEARNING_RATE = 0.001

filepath = '../saved_network_MOFJSP'
TIMESTAMP = time.strftime("%m-%d-%H-%M", time.localtime(time.time()))

def evaluate(env, data, agent, g_pool_step, pref, device):
    # env = FJSP(configs.n_j, configs.n_m)
    adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(data)

    env_mask_mch = torch.from_numpy(mask_mch).to(device)
    env_dur = torch.from_numpy(dur).float().to(device)
    pool = None
    while True:
        adj_temp = torch.from_numpy(adj)
        env_adj = aggr_obs(adj_temp.to(device).to_sparse(), configs.n_j * configs.n_m)
        env_fea = torch.from_numpy(fea).float().to(device)
        env_fea = env_fea.reshape(-1, env_fea.size(-1))
        env_candidate = torch.from_numpy(candidate).long().to(device)
        env_mask = torch.from_numpy(mask).to(device)
        env_mch_time = torch.from_numpy(mch_time).float().to(device)
        # env_job_time = torch.from_numpy(np.copy(job_time)).float().to(device)
        with torch.no_grad():
            action, a_idx, log_a, action_node, _, mask_mch_action, hx = agent.policy_job(x=env_fea,
                                                                                         graph_pool=g_pool_step,
                                                                                         padded_nei=None,
                                                                                         adj=env_adj,
                                                                                         candidate=env_candidate,
                                                                                         mask=env_mask,
                                                                                         mask_mch=env_mask_mch,
                                                                                         dur=env_dur,
                                                                                         a_index=0,
                                                                                         old_action=0,
                                                                                         mch_pool=pool,
                                                                                         old_policy=True,
                                                                                         T=1,
                                                                                         greedy=True)

            pi_mch, pool = agent.policy_mch(action_node, hx, mask_mch_action, env_mch_time)

        _, mch_a = pi_mch.squeeze(-1).max(1)

        adj, fea, reward, done, candidate, mask, job, _, mch_time, job_time = env.step(action.cpu().numpy(), mch_a)
        if env.done_batch.all():
            reward = pref[0].cpu() * env.schedules_batch[:, :, 3].max(-1)[0] + \
                     pref[1].cpu() * env.machines_batch[0].sum(-1)    + \
                     pref[2].cpu() * env.machines_batch[0].max(-1)
                     # pref[1] * env.machines_batch.max(-1)
                     # weight[1] * env.machines_batch[0].sum(-1)
            reward = -reward.item()
            break
    return reward

def sample_noise(agent):
    actor_job = agent.policy_job
    actor_mch = agent.policy_mch
    actor_job_pos = []
    actor_job_neg = []
    for p in actor_job.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise).to(device)
        actor_job_pos.append(noise_t)
        actor_job_neg.append(-noise_t)
    actor_mch_pos = []
    actor_mch_neg = []
    for p in actor_mch.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise).to(device)
        actor_mch_pos.append(noise_t)
        actor_mch_neg.append(-noise_t)
    return actor_job_pos, actor_job_neg, actor_mch_pos, actor_mch_neg


def eval_with_noise(data, agent, actor_job_noise, actor_mch_noise, g_pool_step, pref):
    actor_job = agent.policy_job
    actor_mch = agent.policy_mch
    old_params_actor_job = actor_job.state_dict()
    old_params_actor_mch = actor_mch.state_dict()
    for p, p_n in zip(actor_job.parameters(), actor_job_noise):
        p.data += NOISE_STD * p_n
    for p, p_n in zip(actor_mch.parameters(), actor_mch_noise):
        p.data += NOISE_STD * p_n
    r = evaluate(data, agent, g_pool_step, pref)
    actor_job.load_state_dict(old_params_actor_job)
    actor_mch.load_state_dict(old_params_actor_mch)
    return r

def train_step(agent, actor_job_noise, actor_mch_noise, batch_reward, pref):
    actor_job = agent.policy_job
    actor_mch = agent.policy_mch

    norm_reward = np.array(batch_reward)
    norm_reward -= np.mean(norm_reward)
    s = np.std(norm_reward)
    if abs(s) > 1e-6:
        norm_reward /= s
    norm_reward = torch.from_numpy(norm_reward)
    weighted_noise = None
    for noise, reward in zip(actor_job_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n
    #cm_updates = []
    for p, p_update in zip(actor_job.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * NOISE_STD)
        p.data += LEARNING_RATE * update

    weighted_noise = None
    for noise, reward in zip(actor_mch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n
    for p, p_update in zip(actor_mch.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * NOISE_STD)
        p.data += LEARNING_RATE * update

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # TODO: generate pre
    # np.random.seed(600)
    # a = np.random.uniform(low=1.0, high=101.0, size=(3,))
    # b = a.sum()
    # weight = a / b
    # weight = torch.tensor(weight)
    weitht = torch.tensor([1, 0, 0])
    weight = weitht/weitht.sum()
    print("objective weight: ", weight)

    record = 0

    train_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, MAX_BATCH_EPISODES * MAX_BATCH_STEPS, 400)
    validat_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, 128, 400)
    valid_loader = DataLoader(validat_dataset, batch_size=configs.batch_size)

    data_loader = iter(train_dataset)

    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size(
                                 [1, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                             n_nodes=configs.n_j * configs.n_m,
                             device=device)
    setup_seed(200) # origin 200

    agent = Policy(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
               num_layers=configs.num_layers,
               neighbor_pooling_type=configs.neighbor_pooling_type,
               input_dim=configs.input_dim,
               hidden_dim=configs.hidden_dim,
               num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
               num_mlp_layers_actor=configs.num_mlp_layers_actor,
               hidden_dim_actor=configs.hidden_dim_actor,
               num_mlp_layers_critic=configs.num_mlp_layers_critic,
               hidden_dim_critic=configs.hidden_dim_critic,
               pref_dim = 2)

    agent.policy_job.eval()
    agent.policy_mch.eval()

    vali_list = []
    hv_list = []
    score_list = []
    step_list = []
    seed_idx = 0
    step_idx = 1
    # validation_log = validate(valid_loader, configs.batch_size, agent.policy_job, agent.policy_mch, weight).mean()
    # step_list.append(0)
    # vali_list.append(validation_log.item())
    # print('Step %d The validation quality is: %d' % (0, validation_log))

    for step_idx in range(1, MAX_BATCH_STEPS):
        actor_job_noise = []
        actor_mch_noise = []
        batch_reward = []
        batch_steps = 0
        # # TODO: modify the decoder model and assign function, Done
        pref = torch.rand([2])
        pref = pref / torch.sum(pref)

        agent.policy_job.assign(pref)
        agent.policy_mch.assign(pref)

        for _ in range(MAX_BATCH_EPISODES):
            # TODO: modify sample_noise function, Done
            actor_job_pos, actor_job_neg, actor_mch_pos, actor_mch_neg = sample_noise(agent)
            actor_job_noise.append(actor_job_pos)
            actor_job_noise.append(actor_job_neg)
            actor_mch_noise.append(actor_mch_pos)
            actor_mch_noise.append(actor_mch_neg)

            batch = next(data_loader)
            data = np.expand_dims(batch, axis=0)

            reward = eval_with_noise(data, agent, actor_job_pos, actor_mch_pos, g_pool_step, pref)
            batch_reward.append(reward)
            reward = eval_with_noise(data, agent, actor_job_neg, actor_mch_neg, g_pool_step, pref)
            batch_reward.append(reward)

        # TODO: modify train function, Done
        train_step(agent, actor_job_noise, actor_mch_noise, batch_reward, pref)
        if step_idx==0 or step_idx % 1 == 0:
            # validation_log = validate(valid_loader, configs.batch_size, agent.policy_job, agent.policy_mch, weight).mean()
            hv_score, score1, score2, sum_score = validate2(valid_loader, agent, n_sols=11)
            # TODO: create logger function
            # print('Step %d The validation quality is: %d' % (step_idx, validation_log))
            print("Step id %d, hv is %.2f, sum_score is  %.2f, score1 is %.2f, score2 is %.2f"
                  % (step_idx, hv_score, sum_score, score1, score2))
            step_list.append(step_idx)
            score_list.append(sum_score.item())
            hv_list.append(hv_score.item())
            # TODO: test function
            if record < hv_score:
                epoch_dir = os.path.join(filepath, 'makespan_and_total_time')
                epoch_dir = os.path.join(epoch_dir, TIMESTAMP)
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)
                print("#######Save Step id %d, hv is %d #########" %(step_idx, hv_score))
                job_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_job'))
                machine_savePate = os.path.join(epoch_dir, '{}.pth'.format('policy_mch'))
                torch.save(agent.policy_job.state_dict(), job_savePath)
                torch.save(agent.policy_mch.state_dict(), machine_savePate)
                record = hv_score

    plt.subplot(1, 2, 1)
    plt.plot(step_list, score_list)
    plt.subplot(1, 2, 2)
    plt.plot(step_list, hv_list)
    plt.show()
    print("The best validation quality is %d" %(max(hv_list)))

    # TODO: modify test function
