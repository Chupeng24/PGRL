
from instance_hv_ref import *
import os
import numpy as np
import hvwfg
from mo_utils import *

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

def cal_hv_instances(ins_set_path_list, ins_res_path_list, print_flag=False, print_flag2=True):
    res_dict = dict()
    for file_path in ins_res_path_list:
        df = pd.read_csv(file_path)
        res_dict[file_path[24:-4]] = df

    # Go through all data_path
    for ins_set_path in ins_set_path_list:
        print('#' * 250)
        # GO through instances in data_path
        ins_name_list = [f for f in os.listdir(ins_set_path)]
        ins_name_list.sort()

        hv_dict = dict()
        num_sols_dict = dict()

        #  read the ins result
        for ins_name in ins_name_list:
            ins_res_dict = dict()
            for key in res_dict.keys():
                temp_sols = None
                for i in range(3):
                    list_name = f'{ins_name[:-4]}_{i + 1}'
                    if ins_set_path[16:22] == 'Hurink':
                        list_name = f'{ins_set_path[23]}data_{list_name}'
                    elif ins_set_path[16:18] == 'HU':
                        list_name = f'{ins_set_path[22]}data_{list_name}'

                    if np.all(temp_sols) == None:
                        temp_sols = res_dict[key][list_name].array.reshape(-1, 1)
                    else:
                        temp_sols = np.concatenate((temp_sols, res_dict[key][list_name].array.reshape(-1, 1)), axis=1)
                if np.any(temp_sols) == True:
                    ins_res_dict[key] = temp_sols
                else:
                    continue

            # set hv reference point
            max_value = 0
            for key in ins_res_dict.keys():
                max_value = max(ins_res_dict[key].max(), max_value)
            max_value = max_value+100 if max_value > 100 else max_value+10
            ref = np.full((3, ), max_value)

            # cal instance's hv of each method
            for key in ins_res_dict.keys():
                temp_sols = ins_res_dict[key]
                pop_list = []
                for idx, val in enumerate(temp_sols):
                    if np.all(temp_sols[idx]) == True:
                        pop_list.append(Pop(temp_sols[idx].tolist(), None))

                temp_sols = []
                NDSet = tri_get_pareto_set(pop_list)
                for pop in NDSet[0]:
                    if pop.fitness not in temp_sols:
                        temp_sols.append(pop.fitness)
                n_sols = len(temp_sols)
                temp_sols = np.array(temp_sols)

                temp_hv = hvwfg.wfg(temp_sols.astype(float), ref.astype(float))
                temp_hv = temp_hv / (ref[0] * ref[1] * ref[2])

                if key not in hv_dict.keys():
                    hv_dict[key] = []
                    hv_dict[key].append(temp_hv * 100)
                    num_sols_dict[key] = []
                    num_sols_dict[key].append(n_sols)
                else:
                    hv_dict[key].append(temp_hv * 100)
                    num_sols_dict[key].append(n_sols)


                if print_flag ==  True:
                    print(f"instance-{ins_name}, method-{key}, hv-{temp_hv*100}, num_pare_sols-{n_sols}")

        if print_flag2 == True:
            print('='*180)
            for key in hv_dict.keys():
                print(f'instances set-{ins_set_path}, '
                      f'method-{key}, '
                      f'mean hv-{np.array(hv_dict[key]).mean()}, '
                      f'mean num_sols-{np.array(num_sols_dict[key]).mean()}')






if __name__ == '__main__':
    data_path_list = []
    data_path_list.append(KHB_data_path)
    data_path_list.append(BR_data_path)
    data_path_list.append(BC_data_path)
    data_path_list.append(DP_data_path)
    # data_path_list.append(HU_Rdata_path)
    # data_path_list.append(HU_Edata_path)
    data_path_list.append(HU_Vdata_path)
    # data_path_list.append('FJSP-benchmarks/Hurink_edata')
    # data_path_list.append('FJSP-benchmarks/Hurink_rdata')
    # data_path_list.append('FJSP-benchmarks/Hurink_vdata')
    data_path_list.append('FJSP-benchmarks/HU_la_vdata')

    benchmark_path = '../public_instances_result'
    benchmark_res_path_list = get_imlist(benchmark_path)

    cal_hv_instances(data_path_list, benchmark_res_path_list)