from uniform_instance import FJSPDataset
from FJSP_Env import FJSP
from mb_agg import *
from Params import configs
from torch.utils.data import DataLoader
from policy import Policy
import random
import os
import time
import hvwfg
import pandas as pd
import collections
from torch import multiprocessing as mp
from mo_utils import *
# mpl.style.use('default')


ResItem = collections.namedtuple(
    'ResItem', field_names=['worker_idx', 'pref', 'res'])

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

def worker_func(worker_id, n_j, n_m, filepath, input_queue, res_queue, device):
    configs.n_j=n_j
    configs.n_m=n_m
    ope_nums_of_jobs = np.array([configs.n_m for _ in range(configs.n_j)])
    env = FJSP(configs.n_j, configs.n_m, ope_nums_of_jobs)
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

    # agent.policy_job.assign(pref)
    # agent.policy_mch.assign(pref)

    while True:
        input_data = input_queue.get()
        if input_data is None:
            break
        ins_data = input_data[0]
        pref = input_data[1].to(device)

        agent.policy_job.assign(pref)
        agent.policy_mch.assign(pref)

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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(n_j, n_m, path, ref, n_sols = 15, save_flag=False): # 15, 105, 1035, 10011
    # mp.set_start_method('spawn')
    setup_seed(200)
    mp.set_start_method(method='forkserver', force=True)
    PROCESSES_COUNT = 3

    uniform_weights = None
    if n_sols == 15:
        uniform_weights = torch.Tensor(das_dennis(4, 3))  # 15    # Systematic approach
    if n_sols == 105:
        uniform_weights = torch.Tensor(das_dennis(13, 3))  # 105   # 取组合数来着
    elif n_sols == 1035:
        uniform_weights = torch.Tensor(das_dennis(44, 3))  # 1035
    elif n_sols == 10011:
        uniform_weights = torch.Tensor(das_dennis(140, 3))  # 10011

    res_dict = dict()
    for i, val in enumerate(uniform_weights):
        res_dict[val.to("cpu")] = None

    input_queues = [
        mp.Queue(maxsize=1)
        for _ in range(PROCESSES_COUNT)
    ]
    res_queue = mp.Queue(maxsize=PROCESSES_COUNT)   # TODO: need to change

    net_save_path = path
    workers = []

    gpu_count = torch.cuda.device_count()
    for worker_idx in range(PROCESSES_COUNT):
        if worker_idx == 4:
            gpu_id = 2
        else:
            gpu_id = worker_idx % gpu_count
        device = torch.device("cuda:%d" % (gpu_id))
        p_args = (worker_idx, configs.n_j, configs.n_m, net_save_path, input_queues[worker_idx], res_queue, device)
        proc = mp.Process(target=worker_func, args=p_args)
        proc.start()
        workers.append(proc)

    time.sleep(0.01)

    # print(uniform_weights)
    test_data = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, 100, 200)
    test_loader = DataLoader(test_data, batch_size=configs.batch_size)

    iter = uniform_weights.shape[0]//PROCESSES_COUNT
    iter = iter if uniform_weights.shape[0] % PROCESSES_COUNT == 0 else iter+1
    t_start = time.time()
    for i in range(iter):
        for _, bat in enumerate(test_loader):
            for j, q in enumerate(input_queues):
                q.put((bat, uniform_weights[i*PROCESSES_COUNT + j]))

        count = 0
        while True:
            while not res_queue.empty():
                result = res_queue.get_nowait()
                res_dict[result.pref.to("cpu")] = result.res
                count += 1
            if count == PROCESSES_COUNT:
                break
            time.sleep(0.01)



    total_time = time.time() - t_start
    # print("spend time:", total_time)

    for worker, p_queue in zip(workers, input_queues):
        p_queue.put(None)
        worker.join()

    total_sols = None
    for key in res_dict.keys():
        if np.all(total_sols) == None:
            total_sols = res_dict[key]
        else:
            total_sols = np.concatenate((total_sols, res_dict[key]), axis=0)

    # TODO: origin
    # if save_flag == True:
    #     save_res_dict = dict()
    #     for i in range(test_data.size):
    #         temp_data = total_sols[:, i].reshape((n_sols, 3))
    #         ins_idx = i + 1
    #         for j in range(3):
    #             str_col = "ins_%d_%d" % (ins_idx, j + 1)
    #             save_res_dict[str_col] = temp_data[:, j].tolist()
    #
    #
    # if save_flag == True:
    #     df = pd.DataFrame(save_res_dict)
    #     # df.to_csv('GRL_%dx%d_parallel.csv' % (configs.n_j, configs.n_m))
    #     df.to_csv('GRL_%dx%d_%dx%d_parallel.csv' % (n_j, n_m, configs.n_j, configs.n_m))

    sols_count = n_sols
    print("######################################################################################")
    hv_list = []
    for i in range(test_data.size):
        # temp_sols = total_sols[i*3:(i+1)*3, :].reshape(128, 3)
        temp_sols = total_sols[:, i].reshape(-1, 3)
        temp_hv = hvwfg.wfg(temp_sols.astype(float), ref.astype(float))
        temp_hv = temp_hv / (ref[0] * ref[1] * ref[2])
        hv_list.append(temp_hv)
    hv_ratio = np.array(hv_list).mean()
    print('Run Time(s): {:.4f}'.format(total_time))
    print('Average Run Time(s): {:.4f}'.format(total_time/100))
    print('HV Ratio: {}'.format(hv_ratio))
    print("Number of solutions", sols_count)
    print("######################################################################################")
    sols_count_list = []
    hv_list = []
    for i in range(test_data.size):
        # temp_sols = total_sols[i*3:(i+1)*3, :].reshape(128, 3)
        temp_sols = total_sols[:, i].reshape(-1, 3)
        pop_list = []
        for idx, val in enumerate(temp_sols):
            pref = uniform_weights[idx].to(device)
            pop_list.append(Pop(temp_sols[idx].tolist(), pref))

        temp_sols = []
        NDSet = tri_get_pareto_set(pop_list)
        for pop in NDSet[0]:
            temp_sols.append(pop.fitness)
        sols_count = len(temp_sols)
        sols_count_list.append(sols_count)

        temp_sols = np.array(temp_sols)
        temp_hv = hvwfg.wfg(temp_sols.astype(float), ref.astype(float))
        temp_hv = temp_hv / (ref[0] * ref[1] * ref[2])
        hv_list.append(temp_hv)
    hv_ratio = np.array(hv_list).mean()
    # print('Run Time(s): {:.4f}'.format(total_time))
    print('HV Ratio: {}'.format(hv_ratio))
    print("Number of effective solutions", np.array(sols_count_list).mean())
    print("######################################################################################")
    sols_count_list = []
    hv_list = []
    for i in range(test_data.size):
        # temp_sols = total_sols[i*3:(i+1)*3, :].reshape(128, 3)
        temp_sols = total_sols[:, i].reshape(-1, 3)
        pop_list = []
        for idx, val in enumerate(temp_sols):
            pref = uniform_weights[idx].to(device)
            pop_list.append(Pop(temp_sols[idx].tolist(), pref))

        temp_sols = []
        NDSet = tri_get_pareto_set(pop_list)
        for pop in NDSet[0]:
            if pop.fitness not in temp_sols:
                temp_sols.append(pop.fitness)
        sols_count = len(temp_sols)
        sols_count_list.append(sols_count)

        temp_sols = np.array(temp_sols)
        temp_hv = hvwfg.wfg(temp_sols.astype(float), ref.astype(float))
        temp_hv = temp_hv / (ref[0] * ref[1] * ref[2])
        hv_list.append(temp_hv)
    hv_ratio = np.array(hv_list).mean()
    # print('Run Time(s): {:.4f}'.format(total_time))
    print('HV Ratio: {}'.format(hv_ratio))
    print("Number of effective non repeating solutions", np.array(sols_count_list).mean())

    if save_flag == True:
        spend_time_list = [total_time/100 for _ in range(test_data.size)]
        save_hv_dict = {'hv_score': hv_list, 'spend_time': spend_time_list}
        df = pd.DataFrame(save_hv_dict)
        df.to_csv(f'PGRL-{configs.n_j}x{configs.n_m}.csv')


if __name__ == "__main__":
    # 6x6
    # ref = np.array([2000, 2000, 2000])
    # net_save_path = 'saved_network_MOFJSP/3_obj/07-05-23-01/_72_44.33505371542969'
    # main(path=net_save_path, ref=ref, n_sols = 15, save_flag=True)
    # 10x10
    # ref = np.array([2500, 2500, 2500])
    # net_save_path = 'saved_network_MOFJSP/3_obj/07-06-17-11/_184_25.659656725699996'
    # main(path=net_save_path, ref=ref, n_sols = 15, save_flag=True)
    # 20x20
    # ref = np.array([5000, 5000, 5000])
    # net_save_path = 'saved_network_MOFJSP/3_obj/07-07-10-22/_81_21.15984152106875'
    # main(path=net_save_path, ref=ref, n_sols = 15, save_flag=True)

    # # PSL (10×10) (30x20)
    # ref = np.array([8000, 8000, 8000])
    # net_save_path = 'saved_network_MOFJSP/3_obj/07-06-17-11/_184_25.659656725699996'
    # main(path=net_save_path, ref=ref, n_sols = 15, save_flag=True)
    #
    # # PSL (20×20) (30x20)
    # # ref = np.array([8000, 8000, 8000])
    # # net_save_path = 'saved_network_MOFJSP/3_obj/07-07-10-22/_81_21.15984152106875'
    # # main(path=net_save_path, ref=ref, n_sols = 15, save_flag=True)
    # print("######################################### PSL 6x6 #########################################")
    # configs.n_j = 6
    # configs.n_m = 6
    # ref = np.array([2000, 2000, 2000])
    # net_save_path = 'saved_network_MOFJSP/3_obj/07-05-23-01/_72_44.33505371542969'
    # main(n_j=6, n_m=6, path=net_save_path, ref=ref, n_sols=15, save_flag=True)
    # time.sleep(10)
    #
    # # PSL 10x10
    # print("######################################### PSL 10x10 #########################################")
    # configs.n_j = 10
    # configs.n_m = 10
    # ref = np.array([2500, 2500, 2500])
    # net_save_path = 'saved_network_MOFJSP/3_obj/07-06-17-11/_184_25.659656725699996'
    # main(n_j=10, n_m=10, path=net_save_path, ref=ref, n_sols=15, save_flag=True)
    # time.sleep(10)
    #
    # # PSL 20x20
    # print("######################################### PSL 20x20 #########################################")
    # configs.n_j = 20
    # configs.n_m = 20
    # ref = np.array([5000, 5000, 5000])
    # net_save_path = 'saved_network_MOFJSP/3_obj/07-07-10-22/_81_21.15984152106875'
    # main(n_j=20, n_m=20, path=net_save_path, ref=ref, n_sols=15, save_flag=False)
    # time.sleep(10)
    #
    # # PSL (20×20) (30x20)
    # print("######################################### PSL (20×20) (30x20) #########################################")
    # configs.n_j = 30
    # configs.n_m = 20
    # ref = np.array([8000, 8000, 8000])
    # net_save_path = 'saved_network_MOFJSP/3_obj/07-07-10-22/_81_21.15984152106875'
    # main(n_j=20, n_m=20, path=net_save_path, ref=ref, n_sols=15, save_flag=False)
    # time.sleep(10)
    #
    # # PSL (20×20) (50x20)
    # print("######################################### PSL (20×20) (50x20) #########################################")
    # configs.n_j = 50
    # configs.n_m = 20
    # ref = np.array([12000, 12000, 12000])
    # net_save_path = 'saved_network_MOFJSP/3_obj/07-07-10-22/_81_21.15984152106875'
    # main(n_j=20, n_m=20, path=net_save_path, ref=ref, n_sols=15, save_flag=True)
    # time.sleep(10)
    #
    # # PSL (20×20) (100x20)
    # print("######################################### PSL (20×20) (100x20) #########################################")
    # configs.n_j = 100
    # configs.n_m = 20
    # ref = np.array([20000, 20000, 20000])
    # net_save_path = 'saved_network_MOFJSP/3_obj/07-07-10-22/_81_21.15984152106875'
    # main(n_j=20, n_m=20, path=net_save_path, ref=ref, n_sols=15, save_flag=True)
    # time.sleep(10)
    # TODO: test 15x15
    # print("######################################### PSL 15x15 #########################################")
    # configs.n_j = 15
    # configs.n_m = 15
    # ref = np.array([5000, 5000, 5000])
    # net_save_path = 'saved_network_MOFJSP/3_obj/09-18-00-38/_228_40.178029413784'
    # main(n_j=15, n_m=15, path=net_save_path, ref=ref, n_sols=15, save_flag=True)
    # time.sleep(10)

    # TODO: test time complexity ##############################
    # PSL 6x6
    print("######################################### PSL 6x6 #########################################")
    configs.n_j = 6
    configs.n_m = 6
    ref = np.array([2000, 2000, 2000])
    net_save_path = 'saved_network_MOFJSP/3_obj/6x6'
    main(n_j=6, n_m=6, path=net_save_path, ref=ref, n_sols=15, save_flag=True)
    time.sleep(1)

    # PSL 10x10
    print("######################################### PSL 10x10 #########################################")
    configs.n_j = 10
    configs.n_m = 10
    ref = np.array([2500, 2500, 2500])
    net_save_path = 'saved_network_MOFJSP/3_obj/10x10'
    main(n_j=10, n_m=10, path=net_save_path, ref=ref, n_sols=15, save_flag=True)
    time.sleep(1)

    # PSL 15x15
    print("######################################### PSL 15x15 #########################################")
    configs.n_j = 15
    configs.n_m = 15
    ref = np.array([5000, 5000, 5000])
    net_save_path = 'saved_network_MOFJSP/3_obj/15x15'
    main(n_j=15, n_m=15, path=net_save_path, ref=ref, n_sols=15, save_flag=True)
    time.sleep(1)

    # # PSL 20x20
    print("######################################### PSL 20x20 #########################################")
    configs.n_j = 20
    configs.n_m = 20
    ref = np.array([5000, 5000, 5000])
    net_save_path = 'saved_network_MOFJSP/3_obj/10x10'
    main(n_j=20, n_m=20, path=net_save_path, ref=ref, n_sols = 15, save_flag=True)
    time.sleep(1)
    #
    #
    # # PSL (20×20) (30x20)
    print("######################################### PSL (20×20) (30x20) #########################################")
    configs.n_j = 30
    configs.n_m = 20
    ref = np.array([8000, 8000, 8000])
    net_save_path = 'saved_network_MOFJSP/3_obj/10x10'
    main(n_j=20, n_m=20, path=net_save_path, ref=ref, n_sols = 15, save_flag=True)
    time.sleep(1)

    # PSL (20×20) (30x20)
    print("######################################### PSL (20×20) (40x20) #########################################")
    configs.n_j = 40
    configs.n_m = 20
    ref = np.array([10000, 10000, 10000])
    net_save_path = 'saved_network_MOFJSP/3_obj/10x10'
    main(n_j=20, n_m=20, path=net_save_path, ref=ref, n_sols=15, save_flag=True)
    time.sleep(1)

    # # PSL (20×20) (50x20)
    print("######################################### PSL (20×20) (50x20) #########################################")
    configs.n_j = 50
    configs.n_m = 20
    ref = np.array([12000, 12000, 12000])
    net_save_path = 'saved_network_MOFJSP/3_obj/10x10'
    main(n_j=20, n_m=20, path=net_save_path, ref=ref, n_sols = 15, save_flag=True)
    time.sleep(1)

    # # PSL (20×20) (60x20)
    # print("######################################### PSL (20×20) (60x20) #########################################")
    # configs.n_j = 60
    # configs.n_m = 20
    # ref = np.array([12000, 12000, 12000])
    # net_save_path = 'saved_network_MOFJSP/3_obj/12-15-02-15/_73_25.064545318528005'
    # main(n_j=20, n_m=20, path=net_save_path, ref=ref, n_sols=15, save_flag=False)
    # time.sleep(10)
    #
    # # # PSL (20×20) (70x20)
    # print("######################################### PSL (20×20) (70x20) #########################################")
    # configs.n_j = 70
    # configs.n_m = 20
    # ref = np.array([12000, 12000, 12000])
    # net_save_path = 'saved_network_MOFJSP/3_obj/12-15-02-15/_73_25.064545318528005'
    # main(n_j=20, n_m=20, path=net_save_path, ref=ref, n_sols=15, save_flag=False)
    # time.sleep(10)
    #
    # # # PSL (20×20) (80x20)
    # print("######################################### PSL (20×20) (80x20) #########################################")
    # configs.n_j = 80
    # configs.n_m = 20
    # ref = np.array([12000, 12000, 12000])
    # net_save_path = 'saved_network_MOFJSP/3_obj/12-15-02-15/_73_25.064545318528005'
    # main(n_j=20, n_m=20, path=net_save_path, ref=ref, n_sols=15, save_flag=False)
    # time.sleep(10)
    #
    # # # PSL (20×20) (90x20)
    # print("######################################### PSL (20×20) (90x20) #########################################")
    # configs.n_j = 90
    # configs.n_m = 20
    # ref = np.array([12000, 12000, 12000])
    # net_save_path = 'saved_network_MOFJSP/3_obj/12-15-02-15/_73_25.064545318528005'
    # main(n_j=20, n_m=20, path=net_save_path, ref=ref, n_sols=15, save_flag=False)
    # time.sleep(10)
    #
    # # PSL (20×20) (100x20)
    # print("######################################### PSL (20×20) (100x20) #########################################")
    # configs.n_j = 100
    # configs.n_m = 20
    # ref = np.array([20000, 20000, 20000])
    # net_save_path = 'saved_network_MOFJSP/3_obj/12-15-02-15/_73_25.064545318528005'
    # main(n_j=20, n_m=20, path=net_save_path, ref=ref, n_sols = 15, save_flag=False)
    # time.sleep(10)
