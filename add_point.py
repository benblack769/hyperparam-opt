import json
import sys

def read_json(fname):
    with open(fname) as file:
        return json.loads(file.read())

def add_point(data_fname,new_point):
    old_data = read_json(data_fname)

    old_data['data_points'].append(new_point)
    with open(data_fname,'w') as file:
        json.dump(old_data,file, indent=2)

if __name__ == "__main__":
    assert len(sys.argv) == 3 , "needs two parameters, the data filename, and the data point filename."

    data_fname = sys.argv[1]
    point_fname = sys.argv[2]

    old_data = read_json(data_fname)
    add_point(data_fname,new_point)
