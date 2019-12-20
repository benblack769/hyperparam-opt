#import sklearn
import sys
import json
import numpy as np

def parse_data(data):
    data_specs =  data['dim_ranges']

    ordering = [name for name in data_specs]

    length_scales = data["guassian_params"]["length_scales"]

    xs = []
    ys = []
    for point in data['data_points']:
        ys.append(np.float64(point['reward']))
        cur_xs = np.zeros([len(ordering)])#[None]*len(ordering)
        for vname,val in point['data'].items():
            if vname in length_scales:
                val /= length_scales[vname]
            else:
                # discrete case can be handled by scaling the data to be very large
                val *= 100

            cur_xs[ordering.index(vname)] = val
        xs.append(cur_xs)

    xnpy = np.stack(xs)
    ynpy = np.stack(ys)

    return xnpy,ynpy

def next_point(xs,ys):
    pass

if __name__ == "__main__":
    assert len(sys.argv) == 2 , "needs one parameter, the data filename."
    data = json.load(open(sys.argv[1]))

    xs,ys = parse_data(data)
    print(xs)
    print(ys)
