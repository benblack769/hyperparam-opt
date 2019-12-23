import json
import sys
import yaml

def read_json(fname):
    with open(fname) as file:
        return json.loads(file.read())

def init_optimizer(config_data,data_fname):
    scales = {name: 1e-1 for name,item in config_data['dim_ranges'].items()}
    new_data = config_data
    new_data['data_points'] = []
    new_data['guassian_params'] = {
        "noise": 1e-2,
        "length_scales": scales
    }

    with open(data_fname,'w') as file:
        json.dump(new_data,file, indent=2)

if __name__ == "__main__":
    assert len(sys.argv) == 3 , "needs two parameters, the config filename, and the output data filename."

    config_fname = sys.argv[1]
    data_fname = sys.argv[2]

    config_data = yaml.safe_load(open(config_fname))
    init_optimizer(config_data,data_fname)
