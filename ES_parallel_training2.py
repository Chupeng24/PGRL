from torch import multiprocessing as mp
import numpy as np
from uniform_instance import FJSPDataset
from policy import Policy
import random
import collections
import time
from mb_agg import *
from Params import configs
from validation_optimization_3_obj import validate2
from FJSP_Env import FJSP
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

rng = np.random.default_rng()

MAX_BATCH_EPISODES = 100
MAX_BATCH_STEPS = 300
NOISE_STD = 0.001
LEARNING_RATE = 0.001
# TODO: PROCESSES_COUNT * ITERS_PER_UPDATE should equals to 100
PROCESSES_COUNT = 10
ITERS_PER_UPDATE = 10

RewardsItem = collections.namedtuple(
    'RewardsItem', field_names=['seed', 'pos_reward',
                                'neg_reward'])


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

def sample_noise(agent, device):
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

def eval_with_noise(env, data, agent, actor_job_noise, actor_mch_noise, g_pool_step, pref, device):
    actor_job = agent.policy_job
    actor_mch = agent.policy_mch
    old_params_actor_job = actor_job.state_dict()
    old_params_actor_mch = actor_mch.state_dict()
    for p, p_n in zip(actor_job.parameters(), actor_job_noise):
        p.data += NOISE_STD * p_n
    for p, p_n in zip(actor_mch.parameters(), actor_mch_noise):
        p.data += NOISE_STD * p_n
    r = evaluate(env, data, agent, g_pool_step, pref, device)
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
    # cm_updates = []
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

def worker_func(worker_id, params_queue, rewards_queue,
                device, noise_std):
    ope_nums_of_jobs = np.array([configs.n_m for _ in range(configs.n_j)])
    env = FJSP(configs.n_j, configs.n_m, ope_nums_of_jobs)
    agent = Policy(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
               n_ope=configs.n_j * configs.n_m,
               num_layers=configs.num_layers,
               neighbor_pooling_type=configs.neighbor_pooling_type,
               input_dim=configs.input_dim,
               hidden_dim=configs.hidden_dim,
               num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
               num_mlp_layers_actor=configs.num_mlp_layers_actor,
               hidden_dim_actor=configs.hidden_dim_actor,
               num_mlp_layers_critic=configs.num_mlp_layers_critic,
               hidden_dim_critic=configs.hidden_dim_critic,
               pref_dim = 3,
               device=device)
    agent.policy_job.eval()
    agent.policy_mch.eval()

    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size(
                                 [1, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                             n_nodes=configs.n_j * configs.n_m,
                             device=device)

    while True:
        params = params_queue.get()
        if params is None:
            break
        agent.policy_job.load_state_dict(params[0])
        agent.policy_mch.load_state_dict(params[1])

        pref =  params[2]
        agent.policy_job.assign(pref.to(device))
        agent.policy_mch.assign(pref.to(device))

        for _ in range(ITERS_PER_UPDATE):
            seed = np.random.randint(low=0, high=655350000)
            np.random.seed(seed)
            actor_job_pos, actor_job_neg, actor_mch_pos, actor_mch_neg = sample_noise(agent, device)

            train_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, 1, seed)
            data_loader = iter(train_dataset)

            batch = next(data_loader)
            data = np.expand_dims(batch, axis=0)

            pos_reward = eval_with_noise(env, data, agent, actor_job_pos, actor_mch_pos, g_pool_step, pref, device)
            neg_reward = eval_with_noise(env, data, agent, actor_job_neg, actor_mch_neg, g_pool_step, pref, device)

            rewards_queue.put(RewardsItem(
                seed=seed, pos_reward=pos_reward,
                neg_reward=neg_reward))

if __name__ == "__main__":
    mp.set_start_method('spawn')

    filepath = 'saved_network_MOFJSP'
    TIMESTAMP = time.strftime("%m-%d-%H-%M", time.localtime(time.time()))
    record = 0

    device = torch.device(configs.device)
    print(device)

    # train_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, MAX_BATCH_EPISODES * MAX_BATCH_STEPS, 400)
    validat_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, 128, 200)
    valid_loader = DataLoader(validat_dataset, batch_size=configs.batch_size)

    # g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
    #                          batch_size=torch.Size(
    #                              [1, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
    #                          n_nodes=configs.n_j * configs.n_m,
    #                          device=device)

    # TODO: define env and net
    setup_seed(200) # origin 200

    agent = Policy(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
               n_ope=configs.n_j * configs.n_m,
               num_layers=configs.num_layers,
               neighbor_pooling_type=configs.neighbor_pooling_type,
               input_dim=configs.input_dim,
               hidden_dim=configs.hidden_dim,
               num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
               num_mlp_layers_actor=configs.num_mlp_layers_actor,
               hidden_dim_actor=configs.hidden_dim_actor,
               num_mlp_layers_critic=configs.num_mlp_layers_critic,
               hidden_dim_critic=configs.hidden_dim_critic,
               pref_dim = 3,
               device=device)
    agent.policy_job.eval()
    agent.policy_mch.eval()

    weitht = torch.tensor([1, 1, 1]).to(device)
    weight = weitht/weitht.sum()
    print("objective weight: ", weight)
    agent.policy_job.assign(weight)
    agent.policy_mch.assign(weight)

    t_start_v = time.time()
    hv_score, score1, score2, score3, sum_score = validate2(valid_loader, agent, device, n_sols=15)
    dt_data_v = time.time() - t_start_v
    print("Step id %d, hv is %.2f, sum_score is  %.2f, score1 is %.2f, score2 is %.2f, score3 is %.2f, train time is %.2f, validation time is %.2f"
        % (0, hv_score, sum_score, score1, score2, score3, 0, dt_data_v))


    vali_list = []
    hv_list = []
    score_list = []
    step_list = []
    step_list.append(0)
    score_list.append(sum_score.item())
    hv_list.append(hv_score.item())

    # TODO: define params_queues and rewards_queue
    params_queues = [
        mp.Queue(maxsize=1)
        for _ in range(PROCESSES_COUNT)
    ]

    rewards_queue = mp.Queue(maxsize=ITERS_PER_UPDATE)


    # TODO: define worker and start worker
    workers = []
    gpu_count = torch.cuda.device_count()
    for idx, params_queue in enumerate(params_queues):
        # gpu_id = idx % gpu_count
        # device_temp = torch.device("cuda:%d" % (gpu_id))
        gpu_id = idx % gpu_count
        device = torch.device("cuda:%d" % (gpu_id))
        p_args = (idx, params_queue, rewards_queue,
                device, NOISE_STD)
        proc = mp.Process(target=worker_func, args=p_args)
        proc.start()
        workers.append(proc)

    print("All started!")
    # optimizer = optim.Adam(itertools.chain(agent.policy_job.parameters(),
    #                                        agent.policy_mch.parameters()), lr=LEARNING_RATE)

    for step_idx in range(1, MAX_BATCH_STEPS):
        # pref = (torch.rand([3])).to(device)
        # pref = pref / torch.sum(pref)
        # r = np.random.rand(1)
        # if r < 0.5:
        #     r = np.random.randint(0, 3)
        #     weights = torch.zeros(3).to(device)
        #     weights[r] = 1
        #     weights = weights / torch.sum(weights)
        #     pref = weights

        pref = rng.dirichlet(alpha=(0.2, 0.2, 0.2), size=1)[0]
        pref = torch.from_numpy(pref).float()

        t_start = time.time()

        # broadcasting network params
        job_params = agent.policy_job.state_dict()
        mch_params = agent.policy_mch.state_dict()
        for q in params_queues:
            q.put((job_params, mch_params, pref))
        actor_job_noise = []
        actor_mch_noise = []
        batch_reward = []
        results = 0

        # agent.policy_job.assign(pref)
        # agent.policy_mch.assign(pref)

        while True:
            while not rewards_queue.empty():
                reward = rewards_queue.get_nowait()
                np.random.seed(reward.seed)
                actor_job_pos, actor_job_neg, actor_mch_pos, actor_mch_neg = sample_noise(agent, device)
                actor_job_noise.append(actor_job_pos)
                actor_job_noise.append(actor_job_neg)
                actor_mch_noise.append(actor_mch_pos)
                actor_mch_noise.append(actor_mch_neg)

                batch_reward.append(reward.pos_reward)
                batch_reward.append(reward.neg_reward)

                results += 1
            if results == PROCESSES_COUNT * ITERS_PER_UPDATE:
                break
            time.sleep(0.01)

        dt_data = time.time() - t_start
        train_step(agent, actor_job_noise, actor_mch_noise, batch_reward, pref)

        if step_idx==0 or step_idx % 1 == 0:
            # validation_log = validate(valid_loader, configs.batch_size, agent.policy_job, agent.policy_mch, weight).mean()
            t_start_v = time.time()
            hv_score, score1, score2, score3, sum_score = validate2(valid_loader, agent, device, n_sols=15)
            dt_data_v = time.time() - t_start_v
            # TODO: create logger function, Done
            # print('Step %d The validation quality is: %d' % (step_idx, validation_log))
            print("Step id %d, hv is %.2f, sum_score is  %.2f, score1 is %.2f, score2 is %.2f, score3 is %.2f, train time is %.2f, validation time is %.2f"
                  % (step_idx, hv_score, sum_score, score1, score2, score3, dt_data, dt_data_v))
            step_list.append(step_idx)
            score_list.append(sum_score.item())
            hv_list.append(hv_score.item())
            # TODO: test function
            if record < hv_score:  # TODO: construct a Elite Archive
                step_hv = "_{}_{}".format(step_idx, hv_score.item())
                epoch_dir = os.path.join(filepath, '3_obj')
                epoch_dir = os.path.join(epoch_dir, TIMESTAMP)
                epoch_dir = os.path.join(epoch_dir, step_hv)
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)
                print("#######Save Step id %d, hv is %d #########" %(step_idx, hv_score))
                job_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_job'))
                machine_savePate = os.path.join(epoch_dir, '{}.pth'.format('policy_mch'))
                torch.save(agent.policy_job.state_dict(), job_savePath)
                torch.save(agent.policy_mch.state_dict(), machine_savePate)
                record = hv_score

    for worker, p_queue in zip(workers, params_queues):
        p_queue.put(None)
        worker.join()
    plt.subplot(1, 2, 1)
    plt.plot(step_list, score_list)
    plt.subplot(1, 2, 2)
    plt.plot(step_list, hv_list)
    plt.show()
    print("The best validation quality is %d" % (max(hv_list)))





