import argparse

import numpy as np

from policy import Policy
from mb_agg import *
from mo_utils import *
from utils.fjsp_data_utils import *
from utils.load_data import *
from FJSP_Env import FJSP
from utils.instance_hv_ref import *
import hvwfg
import collections
from torch import multiprocessing as mp

n_sols = 15
if n_sols == 15:
    uniform_weights = torch.Tensor(das_dennis(4, 3))  # 15    # Systematic approach
if n_sols == 105:
    uniform_weights = torch.Tensor(das_dennis(13, 3))  # 105   # 取组合数来着
elif n_sols == 1035:
    uniform_weights = torch.Tensor(das_dennis(44, 3))  # 1035
elif n_sols == 10011:
    uniform_weights = torch.Tensor(das_dennis(140, 3))  # 10011

ResItem = collections.namedtuple(
    'ResItem', field_names=['worker_idx', 'pref', 'res'])

def get_imlist(path):
    return [f for f in os.listdir(path)]

def solve_ins(agent, env, ins_data, number_of_tasks, device, vali_flag=True):
    batch_size = 1
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size(
                                 [1, number_of_tasks, number_of_tasks]),
                             n_nodes=number_of_tasks,
                             device=device)
    adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(ins_data)
    env_mask_mch = torch.from_numpy(mask_mch).to(device)
    env_dur = torch.from_numpy(dur).float().to(device)
    pool = None
    while True:
        adj_temp = torch.from_numpy(adj)
        env_adj = aggr_obs(adj_temp.to(device).to_sparse(), number_of_tasks)
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
            # plt.savefig("./3020_%s.svg"%i, format='svg',dpi=300, bbox_inches='tight')
            # plt.show()
            result = np.zeros((3, batch_size))
            result[0, :] = env.schedules_batch[:, :, 3].max(-1).reshape(1, batch_size)
            result[1, :] = env.machines_batch.sum(-1).reshape(1, batch_size)
            result[2, :] = env.machines_batch.max(-1).reshape(1, batch_size)
            if vali_flag:
                gantt_result = env.validate_gantt()[0]
                if not gantt_result:
                    print("Scheduling Error！！！！！！")
            return result

def solve_ins_multi_obj(agent, ins_data, num_jobs,num_mas, nums_ope_of_jobs, num_opes, device, vali_flag=False):
    total_res = None
    env = FJSP(n_j=num_jobs, n_m=num_mas, ope_nums_of_jobs=nums_ope_of_jobs)
    for i in range(n_sols):
        pref = uniform_weights[i].to(device)
        agent.policy_job.assign(pref)
        agent.policy_mch.assign(pref)
        res = solve_ins(agent, env, ins_data, num_opes, device, vali_flag).reshape(1, 3)
        if np.all(total_res) == None:
            total_res = res
        else:
            total_res = np.concatenate((total_res, res), axis=0)
    # pop_list = []
    # for idx, val in enumerate(total_res):
    #     pref = uniform_weights[idx].to(device)
    #     pop_list.append(Pop(total_res[idx].tolist(), pref))
    #
    # total_res = []
    # NDSet = tri_get_pareto_set(pop_list)
    # for pop in NDSet[0]:
    #     if pop.fitness not in total_res:
    #         total_res.append(pop.fitness)
    # total_res = np.array(total_res)
    # print(total_res)
    return total_res

def sovle_instance_set(ins_set_path, ins_name_list, device, print_flag1=True, print_flag2=True, hv_flag=True, save_dict=None, save_time_dict=None):
    res_list = []
    time_list = []
    for ins_name in ins_name_list:
        ins_path = os.path.join(ins_set_path, ins_name)
        with open(ins_path) as file_object:
            line = file_object.readlines()
        num_jobs, num_mas, num_opes = nums_detec(line)
        matrix_proc_time, nums_ope_of_job = load_fjs(line, num_mas, num_opes)
        print(ins_name, "number of jobs:", num_jobs)
        print(ins_name, "number of mas:", num_mas)
        print(ins_name, "number of operations:", sum(nums_ope_of_job))

        start = time.time()
        # TODO: res return the result
        ins_res_dict = dict()
        iter = uniform_weights.shape[0] // PROCESSES_COUNT
        iter = iter if uniform_weights.shape[0] % PROCESSES_COUNT == 0 else iter + 1
        for i in range(iter):
            for j, q in enumerate(input_queues):
                q.put((matrix_proc_time, num_jobs, num_mas, num_opes, nums_ope_of_job, uniform_weights[i*PROCESSES_COUNT + j]))
            count = 0
            while True:
                while not res_queue.empty():
                    result = res_queue.get_nowait()
                    ins_res_dict[result.pref.to("cpu")] = result.res
                    count += 1
                if count == PROCESSES_COUNT:
                    break
                time.sleep(0.01)
        end = time.time()

        res = None
        for key in ins_res_dict.keys():
            if np.all(res) == None:
                res = ins_res_dict[key].reshape(1, 3)
            else:
                res = np.concatenate((res, ins_res_dict[key].reshape(1, 3)), axis=0)

        filter = True
        temp_res = []
        pop_list = []
        if filter == True:
            for idx, val in enumerate(res):
                pref = None
                pop_list.append(Pop(res[idx].tolist(), pref))

            NDSet = tri_get_pareto_set(pop_list)
            for pop in NDSet[0]:
                if pop.fitness not in temp_res:
                    temp_res.append(pop.fitness)

            for i in range(res.shape[0]-len(temp_res)):
                temp_res.append([0, 0, 0])

            res = np.array(temp_res)

        spend_time = end-start
        if save_dict != None:
            for i in range(3):
                list_name = f'{ins_name[:-4]}_{i+1}'
                if ins_set_path[16:22] == 'Hurink':
                    list_name = f'{ins_set_path[23]}data_{list_name}'
                save_dict[list_name] = res[:, i]
                if i == 0:
                    save_time_dict['ins_name'].append(list_name)
                    save_time_dict['spend_time'].append(spend_time)
        # res_list.append(res)
        time_list.append(spend_time)
        # if print_flag1:
        #     print(f'test on {ins_name} instance,', f'res is {res},', f'spend time is {end-start}')
        if hv_flag:
            ref = [10000, 10000, 10000]
            ref = np.array(ref)
            hv = hvwfg.wfg(res.astype(float), ref.astype(float))
            hv_ratio  = hv / (ref[0] * ref[1] * ref[2])
            res_list.append(hv_ratio*100)
            if print_flag1:
                print(f'test on {ins_name} instance,', 'HV Ratio: {}'.format(hv_ratio * 100), f'spend time is {end - start}')
    if hv_flag and print_flag2:
        print(f'test on {ins_set_path} instance set,', 'mean HV Ratio: {}'.format(np.array(res_list).mean()),
              f'mean spend time is {np.array(time_list).mean()}')
        print("="*160)

def worker_func(worker_id, filepath, input_queue, res_queue, device):

    agent = Policy(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
               n_ope=configs.n_j*configs.n_m,
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

    job_path = './{}.pth'.format('policy_job')
    mch_path = './{}.pth'.format('policy_mch')

    job_path = os.path.join(filepath, job_path)
    mch_path = os.path.join(filepath, mch_path)

    agent.policy_job.load_state_dict(torch.load(job_path))
    agent.policy_mch.load_state_dict(torch.load(mch_path))
    torch.cuda.empty_cache()


    # agent.policy_job.assign(pref)
    # agent.policy_mch.assign(pref)

    while True:
        input_data = input_queue.get()
        if input_data is None:
            break

        ins_data = input_data[0]
        num_jobs = input_data[1]
        num_mas = input_data[2]
        num_opes = input_data[3]
        ope_nums_of_jobs = input_data[4]
        pref = input_data[5].to(device)
        agent.policy_job.assign(pref)
        agent.policy_mch.assign(pref)

        agent.n_j = num_jobs
        agent.n_m = num_mas
        agent.policy_job.n_j = num_jobs
        agent.policy_job.n_m = num_mas
        agent.policy_mch.n_j = num_jobs
        agent.policy_mch.n_m = num_mas
        agent.n_ope = num_opes
        agent.policy_job.n_ope = num_opes

        # ope_nums_of_jobs = np.array([configs.n_m for _ in range(configs.n_j)])
        env = FJSP(num_jobs, num_mas, ope_nums_of_jobs)

        # ins_data = ins_data.numpy()
        batch_size = ins_data.shape[0]
        g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                 batch_size=torch.Size(
                                     [batch_size, num_opes, num_opes]),
                                 n_nodes=num_opes,
                                 device=device)
        adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time, mch_pro_time = env.reset(ins_data)
        env_mask_mch = torch.from_numpy(mask_mch).to(device)
        env_dur = torch.from_numpy(dur).float().to(device)
        pool = None
        while True:
            adj_temp = torch.from_numpy(adj)
            env_adj = aggr_obs(adj_temp.to(device).to_sparse(), num_opes)
            env_fea = torch.from_numpy(fea).float().to(device)
            env_fea = env_fea.reshape(-1, env_fea.size(-1))
            env_candidate = torch.from_numpy(candidate).long().to(device)
            env_mask = torch.from_numpy(mask).to(device)
            env_mch_time = torch.from_numpy(mch_time).float().to(device)
            env_mch_pro_time = torch.from_numpy(mch_pro_time).float().to(device)
            # env_job_time = torch.from_numpy(np.copy(job_time)).float().to(device)
            with torch.no_grad():
                action, a_idx, log_a, action_node, _, mask_mch_action, hx = agent.policy_job( x=env_fea,
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
        res_queue.put(ResItem(worker_idx=worker_id, pref=pref.clone(), res=result))

if __name__ == '__main__':
    # pars hyperparameters
    parser = argparse.ArgumentParser(description='Arguments for solve FJSP')
    params = parser.parse_args()

    # device = torch.device(configs.device)
    # print(device)

    # agent = Policy(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
    #                n_ope=configs.n_j * configs.n_m,
    #                num_layers=configs.num_layers,
    #                neighbor_pooling_type=configs.neighbor_pooling_type,
    #                input_dim=configs.input_dim,
    #                hidden_dim=configs.hidden_dim,
    #                num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
    #                num_mlp_layers_actor=configs.num_mlp_layers_actor,
    #                hidden_dim_actor=configs.hidden_dim_actor,
    #                num_mlp_layers_critic=configs.num_mlp_layers_critic,
    #                hidden_dim_critic=configs.hidden_dim_critic,
    #                pref_dim=3,
    #                device=device)
    # net_save_path = 'saved_network_MOFJSP/3_obj/07-07-10-22/_81_21.15984152106875'
    net_save_path = 'saved_network_MOFJSP/3_obj/10x10'
    # job_path = './{}.pth'.format('policy_job')
    # mch_path = './{}.pth'.format('policy_mch')
    #
    # job_path = os.path.join(net_save_path, job_path)
    # mch_path = os.path.join(net_save_path, mch_path)
    #
    # agent.policy_job.load_state_dict(torch.load(job_path))
    # agent.policy_mch.load_state_dict(torch.load(mch_path))

    # agent.policy_job.eval()
    # agent.policy_mch.eval()

    save_res_dict = {}
    save_res_file_name = 'DRL_public_benchmark.csv'
    save_time_dict = {'ins_name':[], 'spend_time':[]}
    save_time_file_name = 'DRL_public_benchamrk.csv'

    mp.set_start_method(method='forkserver', force=True)
    PROCESSES_COUNT = 3

    res_dict = dict()
    for i, val in enumerate(uniform_weights):
        res_dict[val.to("cpu")] = None
    input_queues = [
        mp.Queue(maxsize=1)
        for _ in range(PROCESSES_COUNT)
    ]
    res_queue = mp.Queue(maxsize=PROCESSES_COUNT)

    workers = []
    gpu_count = torch.cuda.device_count()

    map_gpu_dict = {0:1, 1:2, 2:3,
                    3:1, 4:2, 5:3,
                    6:1, 7:2, 8:3,
                    9:1, 10:2, 11:3,
                    12:1, 13:2, 14:3}

    for worker_idx in range(PROCESSES_COUNT):
        gpu_id = map_gpu_dict[worker_idx]
        device = torch.device("cuda:%d" % (gpu_id))
        pref = uniform_weights[worker_idx].to(device)
        p_args = (worker_idx, net_save_path, input_queues[worker_idx], res_queue, device)
        proc = mp.Process(target=worker_func, args=p_args)
        proc.start()
        workers.append(proc)

    data_path_list = []
    data_path_list.append(KHB_data_path)
    data_path_list.append(BR_data_path)
    data_path_list.append(BC_data_path)
    data_path_list.append(DP_data_path)
    data_path_list.append(HU_Rdata_path)
    data_path_list.append(HU_Edata_path)
    data_path_list.append(HU_Vdata_path)
    data_path_list.append(Fat_data_path)

    for data_path in data_path_list:
        data_name_list = [f for f in os.listdir(data_path)]
        data_name_list.sort()
        sovle_instance_set(ins_set_path=data_path, ins_name_list=data_name_list,
                           device= configs.device, save_dict=save_res_dict, save_time_dict=save_time_dict)

    for worker, p_queue in zip(workers, input_queues):
        p_queue.put(None)
        worker.join()

    if len(save_res_dict.keys()) != None:
        df = pd.DataFrame(save_res_dict)
        df.to_csv(save_res_file_name)
        df = pd.DataFrame(save_time_dict)
        df.to_csv(save_time_file_name)