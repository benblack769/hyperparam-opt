import json
import sys
import yaml

def read_json(fname):
    with open(fname) as file:
        return json.loads(file.read())

def init_small_vals(num_fids,aprox_stev):
    return [aprox_stev/100 for _ in range(num_fids-1)]


def init_optimizer(config_data,data_fname):
    scales = {name: 1e-1 for name,item in config_data['dim_ranges'].items()}
    new_data = dict()
    num_fids =  config_data['num_fidelities']
    aprox_stdev = config_data['aprox_reward_stdev']
    new_data['dim_ranges'] = config_data['dim_ranges']
    new_data['num_fidelities'] = num_fids
    new_data['data_points'] = []
    new_data['guassian_params'] = {
        "noise": 1e-2,
        "length_scales": scales
    }

    new_data['multi_fidelity_params'] = {
        "err_bounds": init_small_vals(num_fids,aprox_stdev),
        "accuracy_targets": init_small_vals(num_fids,aprox_stdev)
    }

    with open(data_fname,'w') as file:
        json.dump(new_data,file, indent=2)

if __name__ == "__main__":
    assert len(sys.argv) == 3 , "needs two parameters, the config filename, and the output data filename."

    config_fname = sys.argv[1]
    data_fname = sys.argv[2]

    config_data = yaml.safe_load(open(config_fname))
    init_optimizer(config_data,data_fname)
