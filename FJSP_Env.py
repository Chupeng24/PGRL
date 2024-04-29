import gym
import numpy as np
from gym.utils import EzPickle
from uniform_instance import override
from Params import configs
from copy import deepcopy


class FJSP(gym.Env, EzPickle):
    def __init__(self,
                 n_j,
                 n_m,
                 ope_nums_of_jobs,
                 ):
        EzPickle.__init__(self)

        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.ope_nums_of_jobs = ope_nums_of_jobs.reshape(self.number_of_jobs,)
        self.number_of_opes = ope_nums_of_jobs.sum()
        # the task id for first column

        self.last_col_sin = np.cumsum(self.ope_nums_of_jobs) - 1
        self.first_col_sin = self.last_col_sin - self.ope_nums_of_jobs + 1
        # print("111111111111111111111111111111")
        # self.getEndTimeLB = calEndTimeLB
        # self.getNghbs = getActionNbghs


    # def done(self):
    #     if np.all(self.partial_sol_sequeence[0] >=0):
    #         return True
    #     return False

    @override
    def step(self, action,mch_a,gantt_plt=None):
        # action is a int 0 - 224 for 15x15 for example
        mch_a = mch_a.cpu().numpy()
        rewards = np.zeros(shape=(self.batch_size, 3))
        self.step_count += 1
        # print(self.step_count)

        # update basic info
        # jobs = action // self.number_of_machines
        jobs = []
        for i in range(self.batch_size):
            jobs.append(np.where(action[i] <= self.last_col)[1][0])
        jobs = np.array(jobs)
        # rel_ope_idx = action % self.number_of_machines
        self.finished_mark[self.batch_idxes, action] = 1

        # self.m[self.batch_idxes, jobs, rel_ope_idx] = mch_a
        self.dur_a = self.proc_times_batch[self.batch_idxes, action, mch_a]
        self.machines_batch[self.batch_idxes, mch_a] += self.dur_a

        self.omega[self.batch_idxes, jobs] = np.where(action==self.last_col[self.batch_idxes, jobs],
                                                      self.omega[self.batch_idxes, jobs],
                                                      self.omega[self.batch_idxes, jobs]+1)

        self.mask[self.batch_idxes, jobs] = np.where(action ==self.last_col[self.batch_idxes, jobs], True, False)

        # dones = None
        self.done_batch = self.mask.all(axis=1)

        #########################################################################
        ###############################cal_start_time############################
        job_ava_time_batch = self.job_time[self.batch_idxes, jobs].reshape(self.batch_size, -1)
        mch_ava_time_batch = self.mch_time[self.batch_idxes, mch_a].reshape(self.batch_size, -1)
        start_time_temp = np.concatenate((job_ava_time_batch, mch_ava_time_batch), axis=1)
        start_time_a = np.max(start_time_temp, axis=1)
        # update 3 matrix
        # np.where(self.mchsStartTimes_2[self.batch_idxes, mch_a] == -configs.high)
        # index_batch = np.array([np.where(self.mchsStartTimes[i, mch_a[i]]==-configs.high)[0][0]
        #                         for i in self.batch_idxes])

        # self.mchsStartTimes[self.batch_idxes, mch_a, index_batch] = start_time_a
        # self.opIDsOnMchs[self.batch_idxes, mch_a, index_batch] = action

        end_time_a = start_time_a + self.dur_a
        # self.mchsEndTimes[self.batch_idxes, mch_a, index_batch] = end_time_a

        # self.temp1[self.batch_idxes, jobs, rel_ope_idx] = end_time_a
        self.schedules_batch[self.batch_idxes, action, 0] = 1
        self.schedules_batch[self.batch_idxes, action, 1] = mch_a
        self.schedules_batch[self.batch_idxes, action, 2] = start_time_a
        self.schedules_batch[self.batch_idxes, action, 3] = end_time_a

        self.mch_time[self.batch_idxes, mch_a] = end_time_a
        self.job_time[self.batch_idxes, jobs] = end_time_a

        self.proc_time_min[self.batch_idxes, action] =  self.dur_a
        self.proc_time_min2[self.batch_idxes, action] = self.dur_a
        self.proc_time_mean[self.batch_idxes, action] = self.dur_a

        # update opes and machines start_time and end time
        # pre_opes = np.where(action - 1 < self.first_col[self.batch_idxes, jobs], self.number_of_opes - 1, action - 1)
        # self.cal_cumul_adj_batch[self.batch_idxes, pre_opes, :] = np.zeros((self.batch_size, self.number_of_opes))
        #
        # is_scheduled = self.finished_mark.reshape(self.batch_size, -1)
        # start_times = self.schedules_batch[self.batch_idxes, :, 2] * is_scheduled
        # un_scheduled = 1 - is_scheduled

        # temp1 = (start_times + self.proc_time_min.reshape(self.batch_size, -1))
        # temp1 = torch.from_numpy(temp1).unsqueeze(1).to("cuda:1")
        # temp2 = torch.from_numpy(self.cal_cumul_adj_batch[self.batch_idxes, :, :]).to("cuda:1")
        # temp3 = torch.from_numpy(un_scheduled).to("cuda:1")
        #
        # estimate_times = torch.bmm(temp1, temp2).squeeze() * temp3

        # estimate_times = np.matmul((start_times + self.proc_time_min2.reshape(self.batch_size, -1))[:, None, :],
        #                            self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze() * un_scheduled
        #
        # self.schedules_batch[self.batch_idxes, :, 2] = start_times + estimate_times
        # self.schedules_batch[self.batch_idxes, :, 3] = self.schedules_batch[self.batch_idxes, :, 2] + self.proc_time_min2.reshape(self.batch_size, -1)
        # self.LBm1 = self.schedules_batch[self.batch_idxes, :, 3].reshape(self.batch_size, self.number_of_jobs, self.number_of_machines)

        # for i in range(self.batch_size):
        #     self.LBm[i] = calEndTimeLBm(self.temp1[i], self.proc_time_min[i])

        self.LBm[self.batch_idxes, action] = end_time_a
        last_ope = self.last_col[self.batch_idxes, jobs]
        for i in self.batch_idxes:
            self.LBm[i, action[i] + 1: last_ope[i] + 1]  = self.proc_time_min2[i, action[i] + 1: last_ope[i] + 1]
            self.LBm[i, action[i]: last_ope[i] + 1] = np.cumsum(self.LBm[i, action[i]: last_ope[i] + 1])

        # for i in range(self.batch_size):
            # last_ope = self.last_col[i][jobs[i]]
            # self.LBm2.reshape(self.batch_size, -1)[i][action[i]] = end_time_a[i]

            # self.LBm2.reshape(self.batch_size, -1)[i][action[i]+1: last_ope+1] \
            #     = self.proc_time_min2.reshape(self.batch_size, -1)[i][action[i]+1: last_ope+1]
            # temp2 = self.LBm2.reshape(self.batch_size, -1)[i][action[i]: last_ope+1]
            # temp2 = np.cumsum(temp2)
            # self.LBm2.reshape(self.batch_size, -1)[i][action[i]: last_ope + 1] \
            #     = np.cumsum(self.LBm2.reshape(self.batch_size, -1)[i][action[i]: last_ope+1])

        feas = np.concatenate((self.LBm.reshape(self.batch_size, self.number_of_opes, 1)/configs.et_normalize_coef,
                                self.finished_mark.reshape(self.batch_size, self.number_of_opes, 1),), -1)

        rewards[:, 0] = -(self.LBm.max(-1) - self.max_endTime)
        rewards[:, 1] = -(self.machines_batch.sum(-1) - self.total_mchtime)
        rewards[:, 2] = -(self.machines_batch.max(-1) - self.cri_mchtime)
        rewards = rewards.tolist()

        pre_opes_mch = np.where(self.mch_cur_opes[self.batch_idxes, mch_a]>=0, self.mch_cur_opes[self.batch_idxes, mch_a], action).astype(int)
        self.adj[self.batch_idxes, action, pre_opes_mch] = 1

        self.mch_cur_opes[self.batch_idxes, mch_a] = action

        self.max_endTime = self.LBm.max(-1)
        self.total_mchtime = self.machines_batch.sum(-1)
        self.cri_mchtime = self.machines_batch.max(-1)

        job_time1 = np.copy(self.job_time)
        mask = np.full(shape=self.mask.shape, fill_value=1, dtype=bool)
        job_time1[self.mask] = float('inf')
        min_avi_time_index = np.where(job_time1 <= job_time1.min(-1).reshape(self.batch_size, 1))
        mask[min_avi_time_index[0], min_avi_time_index[1]] = 0
        mask = self.mask + mask

        return self.adj, feas, rewards, self.done_batch, self.omega, mask, 0, 0, self.mch_time, self.job_time, self.machines_batch

    @override
    def reset(self, data):
        #data (batch_size,n_job,n_mch,n_mch)
        # first_col and last_col
        # shape: (batch_size, num_jobs)
        self.batch_size = data.shape[0]
        self.batch_idxes = np.arange(self.batch_size)
        # self.first_col_sin = np.arange(start=0, stop=self.number_of_opes, step=self.number_of_machines)
        self.first_col = np.tile(self.first_col_sin, (self.batch_size, 1))
        # self.last_col_sin = np.arange(start=self.number_of_machines - 1, stop=self.number_of_opes, step=self.number_of_machines)
        self.last_col = np.tile(self.last_col_sin, (self.batch_size, 1))

        # m is that every operation schedule to which machine
        # shape: (batch_size, num_jobs, num_mas), num_mas is the number of operations per job
        self.step_count = 0
        # self.m = -1 * np.ones((self.batch_size, self.number_of_opes), dtype=int)

        # self.dur is source data (operation process time on machines) of FJSP instance
        # shape: (batch_size, num_jobs, num_mas, num_mas)
        self.proc_times_batch = data.astype(np.single).reshape(self.batch_size, -1, self.number_of_machines) #single单精度浮点数
        # self.dur_cp = deepcopy(self.dur)
        proc_times_batch = deepcopy(self.proc_times_batch)
        dur = deepcopy(self.proc_times_batch)                 # output
        # self.proc_times_batch = self.dur

        # con_nei_up_stream matrix and conj_nei_low_stream matrix.
        # shape: (num_opes, num_opes)
        conj_nei_up_stream = np.eye(self.number_of_opes, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_opes, k=1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col_sin] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[self.last_col_sin] = 0
        self_as_nei = np.eye(self.number_of_opes, dtype=np.single)

        # adj is the Adjacency matrix
        # shape: (batch_size, num_opes, num_opes)
        adj = self_as_nei + conj_nei_up_stream
        self.adj = np.copy(np.broadcast_to(adj, (self.batch_size, self.number_of_opes, self.number_of_opes)))
        # self.adj2 = np.copy(self.adj)

        # cal_cumul_adj = np.zeros((self.number_of_opes, self.number_of_opes))
        # job_idx = -1
        # for i in range(self.number_of_opes):
        #     if i in self.first_col_sin:
        #         job_idx += 1
        #     else:
        #         cal_cumul_adj[self.first_col_sin[job_idx]:i, i] = np.ones((i - self.first_col_sin[job_idx],))
        # self.cal_cumul_adj_batch = np.copy(np.broadcast_to(cal_cumul_adj, (self.batch_size, self.number_of_opes, self.number_of_opes)))

        # initialize mask
        # mask_mch is the mask determine whether ope can be process by mas. shape: (batch_size, num_jobs, num_mas, num_mas)
        # mask is the mask determine whether job is finished.               shape: (batch_size, num_jobs)
        # finished_mark is the mask determine whether ope is done.          shape: (batch_size, num_jobs, num_mas)
        self.mask_mch = np.full(shape=(self.batch_size, self.number_of_opes, self.number_of_machines), fill_value=0,dtype=bool)
        self.mask = np.full(shape=(self.batch_size, self.number_of_jobs), fill_value=0, dtype=bool)
        self.finished_mark = np.zeros((self.batch_size, self.number_of_opes), dtype=int)

        # LBm : min value of Lower bound
        # shape: (batch_size, num_jobs, num_mas)
        input_min = []
        input_mean = []
        for i in range(self.batch_size):

            proc_times_matrix = proc_times_batch[i].reshape(self.number_of_opes, self.number_of_machines).astype(np.single)
            self.mask_mch[i] = np.where(proc_times_matrix <= 0, 1, 0).reshape((self.number_of_opes, self.number_of_machines))

            matrix_proc_time_temp = np.where(proc_times_matrix>0, proc_times_matrix, 1000)
            proc_times_matrix = np.where(proc_times_matrix>0, proc_times_matrix, 0.)
            matrix_ope_ma_adj = np.where(proc_times_matrix>0, 1, 0)

            min_proc = np.min(matrix_proc_time_temp, axis=-1)
            mean_proc = np.sum(proc_times_matrix, axis=-1)/np.sum(matrix_ope_ma_adj, axis=-1)
            input_min.append(min_proc.reshape(self.number_of_opes,))
            input_mean.append(mean_proc.reshape(self.number_of_opes,))

        # self.proc_time_min = np.array(input_min)
        self.proc_time_min = np.array(input_min).reshape(self.batch_size, self.number_of_opes)
        self.proc_time_min2 = np.array(input_min).reshape(self.batch_size, self.number_of_opes)
        self.proc_time_mean = np.array(input_mean).reshape(self.batch_size, self.number_of_opes)
        # self.input_2d = np.concatenate([self.input_min.reshape((self.batch_size, self.number_of_jobs, self.number_of_machines, 1)),
        #                                 self.input_mean.reshape((self.batch_size, self.number_of_jobs, self.number_of_machines, 1))], -1)

        # TODO: need to change
        # self.LBs = np.cumsum(self.input_2d,-2)
        # self.mask_mch = self.mask_mch.reshape((self.batch_size, -1, self.mask_mch.shape[-1]))
        # self.LBm = np.cumsum(self.proc_time_min.reshape(self.batch_size, self.number_of_machines, self.number_of_machines), -1)
        self.LBm = np.zeros((self.batch_size, self.number_of_opes))
        for i in range(self.number_of_jobs):
            self.LBm[self.batch_idxes, self.first_col_sin[i]:self.last_col_sin[i]+1] = \
                np.cumsum(self.proc_time_min[self.batch_idxes, self.first_col_sin[i]:self.last_col_sin[i]+1], axis=-1)

        # machines_batch: Recorded the processing time of each machine
        # machines_batch shape: (batch_size, num_mas)
        self.machines_batch = np.zeros(shape=(self.batch_size, self.number_of_machines))
        self.mch_cur_opes = np.full(shape=(self.batch_size, self.number_of_machines), fill_value=-1)
        # self.mch_time = np.zeros(shape=(self.batch_size, self.number_of_machines))

        '''
        Partial Schedule (state) of jobs/operations, dynamic
            Status
            Allocated machines
            Start time
            End time
        '''
        # schedules_batch shape: (batch_size, num_opes, 4)
        self.schedules_batch = np.zeros(shape=(self.batch_size, self.number_of_opes, 4))
        self.schedules_batch[:, :, 3] = self.LBm.reshape(self.batch_size, -1)
        self.schedules_batch[:, :, 2] = self.schedules_batch[:, :, 3] - \
                                        self.proc_time_min.reshape(self.batch_size, -1)

        # max_endTime, total_mchtime, cri_mchtime shape: (batch_size, )
        self.posRewards = np.zeros(self.batch_size)
        self.initQuality = np.ones(self.batch_size)
        # TODO: need to change
        self.max_endTime = self.LBm.max(-1)
        self.total_mchtime = np.zeros(self.batch_size)
        self.cri_mchtime = np.zeros(self.batch_size)

        # job_time, job available time
        # job_time shape: (batch_size, num_jobs)
        self.job_time = np.zeros((self.batch_size, self.number_of_jobs))

        # feature shape: (batch_size, num_jobs, num_mas, fea_dim)
        fea = np.concatenate((self.LBm.reshape(self.batch_size, self.number_of_opes, 1) / configs.et_normalize_coef,
                              self.finished_mark.reshape(self.batch_size, self.number_of_opes, 1)), axis=-1)

        # initialize feasible omega
        # shape: (batch_size, num_jobs)
        self.omega = self.first_col.astype(np.int64)

        # shape: (batch_size, )
        self.done_batch = self.mask.all(axis=1)

        # mch_time, mch available time
        # mch_time shape: (batch_size, num_mch)
        self.mch_time = np.zeros((self.batch_size, self.number_of_machines))

        # TODO: need to change
        # start time and of operations on machines
        # shape: (batch_size, num_mas, num_tasks)
        # self.mchsStartTimes = -configs.high * np.ones((self.batch_size, self.number_of_machines, self.number_of_opes))
        # self.mchsEndTimes=-configs.high * np.ones((self.batch_size, self.number_of_machines, self.number_of_opes))

        # Ops ID on machines
        # shape: (batch_size, num_mas, num_tasks)
        # self.opIDsOnMchs = -self.number_of_jobs * np.ones((self.batch_size, self.number_of_machines, self.number_of_opes), dtype=np.int32)
        # self.up_mchendtime = np.zeros_like(self.mchsEndTimes)

        # finished time of operation
        # shape: (batch_size, num_job, num_mas)
        # self.temp1 = np.zeros((self.batch_size, self.number_of_jobs, self.number_of_machines))


        self.dur_a = 0


        return self.adj, fea, self.omega, self.mask,self.mask_mch, dur, self.mch_time, self.job_time, self.machines_batch

    def validate_gantt(self):
        '''
        Verify whether the schedule is feasible
        '''
        ma_gantt_batch = [[[] for _ in range(self.number_of_machines)] for __ in range(self.batch_size)]
        for batch_id, schedules in enumerate(self.schedules_batch):
            for i in range(int(self.number_of_opes)):
                step = schedules[i]
                ma_gantt_batch[batch_id][int(step[1])].append([i, step[2].item(), step[3].item()])
        proc_time_batch = self.proc_times_batch

        # Check whether there are overlaps and correct processing times on the machine
        flag_proc_time = 0
        flag_ma_overlap = 0
        flag = 0
        for k in range(self.batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            for i in range(self.number_of_machines):
                ma_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(ma_gantt[i])):
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i])-1):
                        break
                    if ma_gantt[i][j][2]>ma_gantt[i][j+1][1]:
                        flag_ma_overlap += 1
                    if ma_gantt[i][j][2]-ma_gantt[i][j][1] != proc_time[ma_gantt[i][j][0]][i]:
                        flag_proc_time += 1
                    flag += 1

        # Check job order and overlap
        flag_ope_overlap = 0
        for k in range(self.batch_size):
            schedule = self.schedules_batch[k]
            nums_ope = self.number_of_opes
            num_ope_biases = self.first_col[k]
            for i in range(self.number_of_jobs):
                # if int(nums_ope[i]) <= 1:
                #     continue
                for j in range(self.ope_nums_of_jobs[i] - 1):
                    step = schedule[num_ope_biases[i]+j]
                    step_next = schedule[num_ope_biases[i]+j+1]
                    if step[3] > step_next[2]:
                        flag_ope_overlap += 1

        # Check whether there are unscheduled operations
        flag_unscheduled = 0
        for batch_id, schedules in enumerate(self.schedules_batch):
            count = 0
            for i in range(schedules.shape[0]):
                if schedules[i][0]==1:
                    count += 1
            add = 0 if (count == self.number_of_opes) else 1
            flag_unscheduled += add

        if flag_ma_overlap + flag_ope_overlap + flag_proc_time + flag_unscheduled != 0:
            return False, self.schedules_batch
        else:
            return True, self.schedules_batch