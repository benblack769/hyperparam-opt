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
    for x in range(100):
        exec_point = regset.generate_next_point()
        out_value = fid_fns[exec_point.fid_level](exec_point.coord)
        new_point = DataPoint(exec_point.coord,out_value,exec_point.group,exec_point.fid_level)
        regset.add_point(new_point)
        print(json.dumps(new_point.to_dict(),indent=2))
        if not best_point or new_point.reward > best_point.reward:
            best_point = new_point
    print(best_point.to_dict())

if __name__ == "__main__":
    main()
