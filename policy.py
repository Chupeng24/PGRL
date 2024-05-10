from mb_agg import *
from multi_opt_models.PPO_Actor1 import Job_Actor, Mch_Actor
import torch
import time
import torch.nn as nn
from Params import configs

device = torch.device(configs.device)


def initWeights(net, scheme='orthogonal'):
   for e in net.parameters():
      if scheme == 'orthogonal':
         if len(e.size()) >= 2:
            nn.init.orthogonal_(e)

      elif scheme == 'normal':
         nn.init.normal(e, std=1e-2)
      elif scheme == 'xavier':
         nn.init.xavier_normal(e)

class Policy:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_ope,
                 num_layers,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 pref_dim,
                 device,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy_job = Job_Actor(n_j=configs.n_j,
                                    n_m=configs.n_m,
                                    n_ope=n_ope,
                                    num_layers=configs.num_layers,
                                    learn_eps=False,  # todo: change to true
                                    neighbor_pooling_type=configs.neighbor_pooling_type,
                                    input_dim=configs.input_dim,
                                    hidden_dim=configs.hidden_dim,
                                    num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                                    num_mlp_layers_critic=num_mlp_layers_critic,
                                    hidden_dim_critic=hidden_dim_critic,
                                    device=device,
                                    hyper_input_dim=pref_dim)
        self.policy_mch = Mch_Actor(n_j=configs.n_j,
                                    n_m=configs.n_m,
                                    num_layers=configs.num_layers,
                                    learn_eps=True,
                                    neighbor_pooling_type=configs.neighbor_pooling_type,
                                    input_dim=configs.input_dim,
                                    hidden_dim=configs.hidden_dim,
                                    num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                                    device=device,
                                    hyper_input_dim=pref_dim)

        # self.policy_old_job = deepcopy(self.policy_job)
        # self.policy_old_mch = deepcopy(self.policy_mch)
        #
        # self.policy_old_job.load_state_dict(self.policy_job.state_dict())
        # self.policy_old_mch.load_state_dict(self.policy_mch.state_dict())

        # self.job_optimizer = torch.optim.Adam(self.policy_job.parameters(), lr=lr)
        # self.mch_optimizer = torch.optim.Adam(self.policy_mch.parameters(), lr=lr)
        #
        # self.job_scheduler = torch.optim.lr_scheduler.StepLR(self.job_optimizer,
        #                                                  step_size=configs.decay_step_size,
        #                                                  gamma=configs.decay_ratio)
        # self.mch_scheduler = torch.optim.lr_scheduler.StepLR(self.mch_optimizer,
        #                                                  step_size=configs.decay_step_size,
        #                                                  gamma=configs.decay_ratio)
        #
        #
        # self.MSE = nn.MSELoss()

#     def update(self,  memories, epoch, obj_weight):
#         '''self.policy_job.train()
#         self.policy_mch.train()'''
#         vloss_coef = configs.vloss_coef
#         ploss_coef = configs.ploss_coef
#         entloss_coef = configs.entloss_coef
#         rewards_all_env = []
#
#         for i in range(configs.batch_size):
#             rewards1 = []
#             discounted_reward = 0
#             for reward, is_terminal in zip(reversed((memories.r_mb[0][i][:,0]).tolist()),
#                                            reversed(memories.done_mb[0][i].tolist())):
#                 if is_terminal:
#                     discounted_reward = 0
#                 discounted_reward = reward + (self.gamma * discounted_reward)
#                 rewards1.insert(0, discounted_reward)
#             rewards1 = torch.tensor(rewards1, dtype=torch.float).to(device)
#             rewards1 = (rewards1 - rewards1.mean()) / (rewards1.std() + 1e-8)
#
#             rewards2 = []
#             discounted_reward = 0
#             for reward, is_terminal in zip(reversed((memories.r_mb[0][i][:,1]).tolist()),
#                                            reversed(memories.done_mb[0][i].tolist())):
#                 if is_terminal:
#                     discounted_reward = 0
#                 discounted_reward = reward + (self.gamma * discounted_reward)
#                 rewards2.insert(0, discounted_reward)
#             rewards2 = torch.tensor(rewards2, dtype=torch.float).to(device)
#             rewards2 = (rewards2 - rewards2.mean()) / (rewards2.std() + 1e-8)
#
#             rewards3 = []
#             discounted_reward = 0
#             # update function modify
#             for reward, is_terminal in zip(reversed((memories.r_mb[0][i][:,2]).tolist()),
#                                            reversed(memories.done_mb[0][i].tolist())):
#                 if is_terminal:
#                     discounted_reward = 0
#                 discounted_reward = reward + (self.gamma * discounted_reward)
#                 rewards3.insert(0, discounted_reward)
#             rewards3 = torch.tensor(rewards3, dtype=torch.float).to(device)
#             rewards3 = (rewards3 - rewards3.mean()) / (rewards3.std() + 1e-8)
#
#             sum_reward = obj_weight[0] * rewards1 + obj_weight[1] * rewards2 + obj_weight[2] * rewards3
#             rewards_all_env.append(sum_reward)
#
#         rewards_all_env = torch.stack(rewards_all_env, 0)
#         for _ in range(configs.k_epochs):
#             loss_sum = 0
#             vloss_sum = 0
#             g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
#                                      batch_size=torch.Size(
#                                          [configs.batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
#                                      n_nodes=configs.n_j * configs.n_m,
#                                      device=device)
#
#             job_log_prob = []
#             mch_log_prob = []
#             val = []
#             mch_a =None
#             last_hh = None
#             entropies = []
#             job_entropy = []
#             mch_entropies = []
#             job_scheduler = LambdaLR(self.job_optimizer, lr_lambda=lambda f: 0.96 ** epoch)
#             mch_scheduler = LambdaLR(self.mch_optimizer, lr_lambda=lambda f: 0.96 ** epoch)
#             job_log_old_prob = memories.job_logprobs[0]
#             mch_log_old_prob = memories.mch_logprobs[0]
#             env_mask_mch = memories.mask_mch[0]
#             env_dur = memories.dur[0]
#             first_task = memories.first_task[0]
#             pool=None
#             for i in range(len(memories.fea_mb)):
#                 env_fea = memories.fea_mb[i]
#                 env_adj = memories.adj_mb[i]
#                 env_candidate = memories.candidate_mb[i]
#                 env_mask = memories.mask_mb[i]
#
#
#                 a_index = memories.a_mb[i]
#                 env_mch_time = memories.mch_time[i]
#
#                 old_action = memories.action[i]
#                 old_mch = memories.mch[i]
#
#                 a_entropy, v, log_a, action_node, _, mask_mch_action, hx = self.policy_job(x=env_fea,
#                                                                                            graph_pool=g_pool_step,
#                                                                                            padded_nei=None,
#                                                                                            adj=env_adj,
#                                                                                            candidate=env_candidate
#                                                                                            , mask=env_mask
#                                                                                            , mask_mch=env_mask_mch
#                                                                                            , dur=env_dur
#                                                                                            , a_index=a_index
#                                                                                            , old_action=old_action
#                                                                                            ,mch_pool=pool
#                                                                                            , old_policy=False
#                                                                                            )
#                 pi_mch,pool = self.policy_mch(action_node, hx, mask_mch_action, env_mch_time,mch_a,last_hh,policy=True)
#                 val.append(v)
#                 dist = Categorical(pi_mch)
#                 log_mch = dist.log_prob(old_mch)
#                 mch_entropy = dist.entropy()
#
#                 job_entropy.append(a_entropy)
#                 mch_entropies.append(mch_entropy)
#                 # entropies.append((mch_entropy+a_entropy))
#
#                 job_log_prob.append(log_a)
#                 mch_log_prob.append(log_mch)
#
#             job_log_prob, job_log_old_prob = torch.stack(job_log_prob, 0).permute(1, 0), torch.stack(job_log_old_prob,
#                                                                                                      0).permute(1, 0)
#             mch_log_prob, mch_log_old_prob = torch.stack(mch_log_prob, 0).permute(1, 0), torch.stack(mch_log_old_prob,
#                                                                                                      0).permute(1, 0)
#             val = torch.stack(val, 0).squeeze(-1).permute(1, 0)
#             job_entropy = torch.stack(job_entropy, 0).permute(1, 0)
#             mch_entropies = torch.stack(mch_entropies, 0).permute(1, 0)
#
#             job_loss_sum = 0
#             job_v_loss_sum = 0
#             mch_loss_sum = 0
#             mch_v_loss_sum = 0
#             for j in range(configs.batch_size):
#                 job_ratios = torch.exp(job_log_prob[j] - job_log_old_prob[j].detach())
#                 mch_ratios = torch.exp(mch_log_prob[j] - mch_log_old_prob[j].detach())
#                 advantages = rewards_all_env[j] - val[j].detach()
#                 advantages = adv_normalize(advantages)
#                 job_surr1 = job_ratios * advantages
#                 job_surr2 = torch.clamp(job_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
#                 job_v_loss = self.MSE(val[j], rewards_all_env[j])
#                 job_loss = -1*torch.min(job_surr1, job_surr2) + 0.5*job_v_loss - 0.01 * job_entropy[j]
#                 job_loss_sum = job_loss_sum + job_loss
#                 job_v_loss_sum =  job_v_loss_sum + job_v_loss
#
#                 mch_surr1 = mch_ratios * advantages
#                 mch_surr2 = torch.clamp(mch_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
#                 #mch_v_loss = self.MSE(val[j], rewards_all_env[j])
#                 mch_loss = -1*torch.min(mch_surr1, mch_surr2) - 0.01 * mch_entropies[j] # + 0.5*job_v_loss
#                 mch_loss_sum = mch_loss_sum + mch_loss
#                 mch_v_loss_sum = mch_v_loss_sum + job_v_loss
#
#             # take gradient step
#             # loss_sum = torch.stack(loss_sum,0)
#             # v_loss_sum = torch.stack(v_loss_sum,0)
#             self.job_optimizer.zero_grad()
#             job_loss_sum.mean().backward(retain_graph=True)
#             self.job_optimizer.step()
#             # scheduler.step()
#             # Copy new weights into old policy:
#             self.policy_old_job.load_state_dict(self.policy_job.state_dict())
#             if configs.decayflag:
#                 self.job_scheduler.step()
#
#             self.mch_optimizer.zero_grad()
#             mch_loss_sum.mean().backward(retain_graph=True)
#             self.mch_optimizer.step()
#             # scheduler.step()
#             # Copy new weights into old policy:
#             self.policy_old_mch.load_state_dict(self.policy_mch.state_dict())
#             if configs.decayflag:
#                 self.mch_scheduler.step()
#             # print(configs.lr)
#
#             return job_loss_sum.mean().item(), mch_loss_sum.mean().item()
#
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
# def main(epochs):
#     from uniform_instance import FJSPDataset
#     from FJSP_Env import FJSP
#
#     np.random.seed(700)
#     a = np.random.uniform(low=1.0, high=101.0, size=(3,))
#     b = a.sum()
#     weight = a/b
#     weight = torch.tensor(weight)
#     print("objective weight: ", weight)
#
#     log = []
#
#     g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
#                              batch_size=torch.Size(
#                                  [configs.batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
#                              n_nodes=configs.n_j * configs.n_m,
#                              device=device)
#     setup_seed(400)
#     ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
#               n_j=configs.n_j,
#               n_m=configs.n_m,
#               num_layers=configs.num_layers,
#               neighbor_pooling_type=configs.neighbor_pooling_type,
#               input_dim=configs.input_dim,
#               hidden_dim=configs.hidden_dim,
#               num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
#               num_mlp_layers_actor=configs.num_mlp_layers_actor,
#               hidden_dim_actor=configs.hidden_dim_actor,
#               num_mlp_layers_critic=configs.num_mlp_layers_critic,
#               hidden_dim_critic=configs.hidden_dim_critic)
#     train_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, configs.num_ins, 300)
#     validat_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, 128, 200)
#
#     data_loader = DataLoader(train_dataset, batch_size=configs.batch_size)
#     valid_loader = DataLoader(validat_dataset, batch_size=configs.batch_size)
#     '''ppo.policy_old_job.to(device)
#     ppo.policy_old_mch.to(device)'''
#
#     vali_list = []
#     validation_log = validate(valid_loader, configs.batch_size, ppo.policy_job, ppo.policy_mch, weight).mean()
#     print('The validation quality is:', validation_log)
#     vali_list.append(validation_log.item())
#
#     record = 1000000
#     for epoch in range(epochs):
#         memory = Memory()
#         ppo.policy_old_job.train()
#         ppo.policy_old_mch.train()
#
#         times, losses, rewards2, critic_rewards = [], [], [], []
#         start = time.time()
#
#         costs = []
#         losses, rewards, critic_loss = [], [], []
#         for batch_idx, batch in enumerate(data_loader):
#             env = FJSP(configs.n_j, configs.n_m)
#             data = batch.numpy()
#             # TODO: pre definition, Done
#             pref = torch.rand([2])
#             pref = pref / torch.sum(pref)
#             # TODO: conduct the assign, that is hypermodel function and its forward function
#             # ppo.policy_job.assign(pref)
#             # ppo.policy_mch.assign(pref)
#
#
#             adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(data)
#
#             job_log_prob = []
#             mch_log_prob = []
#             r_mb = []
#             done_mb = []
#             first_task = []
#             pretask = []
#             j = 0
#             mch_a = None
#             last_hh = None
#             pool = None
#             ep_rewards = - env.initQuality
#             env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
#             env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
#             while True:
#
#                 env_adj = aggr_obs(deepcopy(adj).to(device).to_sparse(), configs.n_j * configs.n_m)
#                 env_fea = torch.from_numpy(np.copy(fea)).float().to(device)
#                 env_fea = deepcopy(env_fea).reshape(-1, env_fea.size(-1))
#                 env_candidate = torch.from_numpy(np.copy(candidate)).long().to(device)
#
#                 env_mask = torch.from_numpy(np.copy(mask)).to(device)
#                 env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(device)
#                 # env_job_time = torch.from_numpy(np.copy(job_time)).float().to(device)
#                 with torch.no_grad():
#                     action, a_idx, log_a, action_node, _, mask_mch_action, hx = ppo.policy_old_job(x=env_fea,
#                                                                                                graph_pool=g_pool_step,
#                                                                                                padded_nei=None,
#                                                                                                adj=env_adj,
#                                                                                                candidate=env_candidate
#                                                                                                , mask=env_mask
#
#                                                                                                , mask_mch=env_mask_mch
#                                                                                                , dur=env_dur
#                                                                                                , a_index=0
#                                                                                                , old_action=0
#                                                                                                , mch_pool=pool
#                                                                                                )
#
#                     pi_mch,pool = ppo.policy_old_mch(action_node, hx, mask_mch_action, env_mch_time,mch_a,last_hh)
#                 # print(action,mch_a)
#                     mch_a, log_mch = select_action2(pi_mch)
#                 job_log_prob.append(log_a)
#
#                 # print(action[0].item(),mch_a[0].item())
#                 mch_log_prob.append(log_mch)
#
#                 memory.mch.append(mch_a)
#                 memory.pre_task.append(pretask)
#                 memory.adj_mb.append(env_adj)
#                 memory.fea_mb.append(env_fea)
#                 memory.candidate_mb.append(env_candidate)
#                 memory.action.append(deepcopy(action))
#                 memory.mask_mb.append(env_mask)
#                 memory.mch_time.append(env_mch_time)
#                 memory.a_mb.append(a_idx)
#
#                 adj, fea, reward, done, candidate, mask, job, _, mch_time, job_time = env.step(action.cpu().numpy(),
#                                                                                                mch_a)
#                 # ep_rewards += reward
#                 # TODO: modify ep_rewards
#                 r_mb.append(deepcopy(reward))
#                 done_mb.append(deepcopy(done))
#
#                 j += 1
#                 if env.done():
#                     break
#             memory.dur.append(env_dur)
#             memory.mask_mch.append(env_mask_mch)
#             memory.first_task.append(first_task)
#             memory.job_logprobs.append(job_log_prob)
#             memory.mch_logprobs.append(mch_log_prob)
#             memory.r_mb.append(torch.tensor(r_mb).float().permute(1, 0, 2))
#             memory.done_mb.append(torch.tensor(done_mb).float().permute(1, 0))
#             # -------------------------------------------------------------------------------------
#             # ep_rewards -= env.posRewards
#             # -------------------------------------------------------------------------------------
#             loss, v_loss = ppo.update(memory,batch_idx, weight)
#             memory.clear_memory()
#             mean_reward = np.mean(ep_rewards)
#             log.append([batch_idx, mean_reward])
#
#
#             if batch_idx % 100 == 0:
#                 file_writing_obj = open(
#                     './' + 'log_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(
#                         configs.high) + '.txt', 'w')
#                 file_writing_obj.write(str(log))
#
#             rewards.append(np.mean(ep_rewards).item())
#             losses.append(loss)
#             critic_loss.append(v_loss)
#
#             cost = env.mchsEndTimes.max(-1).max(-1)
#             costs.append(cost.mean())
#             step = 20
#
#             filepath = 'saved_network'
#             if (batch_idx + 1) % step  == 0 :
#                 end = time.time()
#                 times.append(end - start)
#                 start = end
#                 mean_loss = np.mean(losses[-step:])
#                 mean_reward = np.mean(costs[-step:])
#                 critic_losss = np.mean(critic_loss[-step:])
#
#                 filename = 'FJSP_{}'.format('J'+str(configs.n_j)+'M'+str(configs.n_m))
#                 filepath = os.path.join(filepath, filename)
#                 epoch_dir = os.path.join(filepath, '%s_%s' % (100, batch_idx))
#                 if not os.path.exists(epoch_dir):
#                     os.makedirs(epoch_dir)
#                 job_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_job'))
#                 machine_savePate = os.path.join(epoch_dir, '{}.pth'.format('policy_mch'))
#
#                 torch.save(ppo.policy_job.state_dict(), job_savePath)
#                 torch.save(ppo.policy_mch.state_dict(), machine_savePate)
#
#                 print('  Batch %d/%d, reward: %2.3f, loss: %2.4f,critic_loss:%2.4f,took: %2.4fs' %
#                       (batch_idx, len(data_loader), mean_reward, mean_loss, critic_losss,
#                        times[-1]))
#
#                 t4 = time.time()
#                 validation_log = validate(valid_loader, configs.batch_size, ppo.policy_job, ppo.policy_mch, weight).mean()
#                 vali_list.append(validation_log.item())
#                 if validation_log<record:
#                     epoch_dir = os.path.join(filepath, 'best_value100')
#                     if not os.path.exists(epoch_dir):
#                         os.makedirs(epoch_dir)
#                     job_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_job'))
#                     machine_savePate = os.path.join(epoch_dir, '{}.pth'.format('policy_mch'))
#                     torch.save(ppo.policy_job.state_dict(), job_savePath)
#                     torch.save(ppo.policy_mch.state_dict(), machine_savePate)
#                     record = validation_log
#
#                 print('The validation quality is:', validation_log)
#                 file_writing_obj1 = open(
#                     './' + 'vali_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(
#                         configs.high) + '.txt', 'w')
#                 file_writing_obj1.write(str(validation_log))
#                 t5 = time.time()
#         print("the best validation result: ", record)
#         plt.plot(range(len(vali_list)), vali_list)
#         plt.show()
#         np.savetxt('./N_%s_M%s_u100'%(configs.n_j,configs.n_m),costs,delimiter="\n")



if __name__ == '__main__':
    total1 = time.time()
    main(1)
    total2 = time.time()

    #print(total2 - total1)