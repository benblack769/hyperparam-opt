from regressor_set import RegressorSet,DataPoint
import yaml
from borehole import borehole, low_fidelity_borehole
import json

fid_fns = [low_fidelity_borehole,borehole]

def main():
    config_fname = "borehole.yaml"

    with open(config_fname) as yaml_file:
        config = yaml.safe_load(yaml_file)
    print(config)
    regset = RegressorSet(config)
    best_point = None
    for x in range(300):
        exec_point = regset.generate_next_point()
        out_value = fid_fns[exec_point.fid_level](exec_point.coord)
        new_point = exec_point.to_data(out_value)
        regset.add_point(new_point)
        print(new_point.fid_level,"\t",new_point.reward)
        #print(json.dumps(new_point.to_dict(),indent=2))
        if not best_point or new_point.reward > best_point.reward:
            best_point = new_point
    print(best_point.to_dict())

if __name__ == "__main__":
    main()
