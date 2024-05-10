from mb_agg import *
from FJSP_Env import FJSP
from mb_agg import g_pool_cal
import numpy as np
import torch
from Params import configs
import hvwfg
from policy import Policy
import os
from uniform_instance import FJSPDataset
from torch.utils.data import DataLoader

def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)

def solve_batch_ins(ins_data, agent, pref):
    device = torch.device(configs.device)
    ope_nums_of_jobs = np.array([configs.n_m for _ in range(configs.n_j)])
    env = FJSP(configs.n_j, configs.n_m, ope_nums_of_jobs)
    ins_data = ins_data.numpy()
    batch_size = ins_data.shape[0]
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size(
                                 [batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                             n_nodes=configs.n_j * configs.n_m,
                             device=device)
    adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time, mch_pro_time = env.reset(ins_data)
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
        env_mch_pro_time = torch.from_numpy(mch_pro_time).float().to(device)
        # env_job_time = torch.from_numpy(np.copy(job_time)).float().to(device)
        # with torch.no_grad():
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
                                                                                     greedy=True)

        pi_mch, pool = agent.policy_mch(action_node, hx, mask_mch_action, env_mch_time, env_mch_pro_time)

        _, mch_a = pi_mch.squeeze(-1).max(1)

        adj, fea, reward, done, candidate, mask, job, _, mch_time, job_time, mch_pro_time = env.step(action.cpu().numpy(), mch_a)
        # rewards += reward

        if env.done_batch.all():
            # plt.savefig("./3020_%s.svg"%i, format='svg',dpi=300, bbox_inches='tight')
            # plt.show()
            result = np.zeros((3, batch_size))
            result[0, :] = env.schedules_batch[:, :, 3].max(-1).reshape(1, batch_size)
            result[1, :] = env.machines_batch.sum(-1).reshape(1, batch_size)
            result[2, :] = env.machines_batch.max(-1).reshape(1, batch_size)
            break
    return result


def validate2(dataloader, agent, device, n_sols=15, if_pref=True, pref=None, ):
    agent.policy_job.eval()
    agent.policy_mch.eval()
    # TODO: process result return 1. sum_weight result; 2. score 1; 3.score 2;
    if if_pref == True:
        pref = torch.tensor([0.3333, 0.3333, 0.3333]).to(device)
        agent.policy_job.assign(pref)
        agent.policy_mch.assign(pref)
        pref = pref.to("cpu").numpy()
        totall_result = np.concatenate([solve_batch_ins(bat, agent, pref) for i, bat in enumerate(dataloader)], 1)
        score1 = totall_result[0, :].mean(-1).item()
        score2 = totall_result[1, :].mean(-1).item()
        score3 = totall_result[2, :].mean(-1).item()
        sum_score = totall_result[0, :].mean(-1) * pref[0] + \
                    totall_result[1, :].mean(-1).item() * pref[1] + \
                    totall_result[2, :].mean(-1).item() * pref[2]

    # TODO: def sols list
    sols = np.zeros([n_sols, 3])
    # TODO: the process to solve
    # x_list = torch.linspace(start=0, end=1, steps=n_sols)
    # y_list = torch.linspace(start=1, end=0, steps=n_sols)
    uniform_weights = None
    if n_sols == 15:
        uniform_weights = torch.Tensor(das_dennis(4, 3))  # 15    # Systematic approach
    if n_sols == 105:
        uniform_weights = torch.Tensor(das_dennis(13,3))  # 105   # 取组合数来着
    elif n_sols == 1035:
        uniform_weights = torch.Tensor(das_dennis(44,3))   # 1035
    elif n_sols == 10011:
        uniform_weights = torch.Tensor(das_dennis(140,3))   # 10011

    # ref = np.array([2000, 2000, 2000]) # 6x6
    # ref = np.array([5000, 5000, 5000])  # 15x15
    ref = np.array([2500, 2500, 2500])  # 10x10
    # ref = np.array([2000, 2000, 2000])
    # ref = np.array([5000, 5000, 5000])  # 20x20
    # ref = np.array([5000, 5000, 5000])  # 10x10
    # ref = np.array([3500, 3500, 3500])
    # ref = np.array([20000, 20000, 20000]) # 15x15 need to improve
    total_sols = None
    for i in range(n_sols):
        pref = uniform_weights[i].to(device)

        agent.policy_job.assign(pref)
        agent.policy_mch.assign(pref)
        totall_result = np.concatenate([solve_batch_ins(bat, agent, pref) for i, bat in enumerate(dataloader)], 1)
        if np.all(total_sols) == None:
            total_sols = totall_result
        else:
            total_sols = np.concatenate((total_sols, totall_result), axis=0)
        sols[i] = totall_result.mean(-1).reshape(1, 3)

    # TODO: change hv cal method
    # ref = np.array([1500, 1500, 1500])

    # hv = hvwfg.wfg(sols.astype(float), ref.astype(float))
    # hv_ratio = hv / (ref[0] * ref[1] * ref[2])

    # print(hv_ratio)

    hv_list = []
    for i in range(dataloader.dataset.size):
        # temp_sols = total_sols[i*3:(i+1)*3, :].reshape(128, 3)
        temp_sols = total_sols[:, i].reshape(-1, 3)
        temp_hv = hvwfg.wfg(temp_sols.astype(float), ref.astype(float))
        temp_hv = temp_hv / (ref[0] * ref[1] * ref[2])
        hv_list.append(temp_hv)
    hv_ratio = np.array(hv_list).mean()

    return hv_ratio * 100, score1, score2, score3, sum_score

if __name__ == "__main__":
    device = torch.device(configs.device)
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
                        pref_dim=3,
                        device=device)

    env = FJSP(configs.n_j, configs.n_m)

    filepath = 'saved_network_MOFJSP'
    filepath = os.path.join(filepath, "3_obj")
    # filepath = os.path.join(filepath, '06-02-22-08')
    # filepath = os.path.join(filepath, '_222_43.22564571523531')
    filepath = os.path.join(filepath, '07-05-23-01')
    filepath = os.path.join(filepath, '_72_44.33505371542969')

    job_path = './{}.pth'.format('policy_job')
    mch_path = './{}.pth'.format('policy_mch')

    job_path = os.path.join(filepath, job_path)
    mch_path = os.path.join(filepath, mch_path)

    agent.policy_job.load_state_dict(torch.load(job_path))
    agent.policy_mch.load_state_dict(torch.load(mch_path))

    agent.policy_job.eval()
    agent.policy_mch.eval()

    weitht = torch.tensor([1, 1, 1]).to(device)
    pref = weitht/weitht.sum()
    agent.policy_job.assign(pref)
    agent.policy_mch.assign(pref)


    validat_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, 128, 200)
    valid_loader = DataLoader(validat_dataset, batch_size=configs.batch_size)

    hv_score, score1, score2, score3, sum_score = validate2(valid_loader, agent, device, n_sols=15)
    print(hv_score)
