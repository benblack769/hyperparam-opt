#import sklearn
import sys
import json
import numpy as np
import math
import sklearn
import sklearn.gaussian_process
import scipydirect

def format_disc_comb(name,val):
    return name+"_"+str(val)

def format_combs(combs):
    return ";".join(sorted(combs))

def all_comb_helper(disc_vals,cur_comb,cur_combs):
    if not disc_vals:
        cur_combs.append(cur_comb)
    else:
        name,num = disc_vals[0]
        for val in range(num):
            all_comb_helper(disc_vals[1:],cur_comb+[format_disc_comb(name,val)],cur_combs)

def all_combinations(disc_vals):
    cur_combs = []
    all_comb_helper(disc_vals,[],cur_combs)
    return [format_combs(v) for v in cur_combs]

def order_vals(data,vals):
    ordering = data['contin_ordering']

    ordered_scales = []
    for name in ordering:
        ordered_scales.append(vals[name])
    return ordered_scales

def get_discrete_vals(data):
    data_specs = data['dim_ranges']
    ordering = data['disc_ordering']
    discrete_vals = []

    for name in ordering:
        specs =  data_specs[name]
        if specs['type'] == 'discrete':
            num_possiblities = data_specs[name]['possibilities']
            discrete_vals.append((name,num_possiblities))
    return discrete_vals

def transform(trans,val):
    if trans['log']:
        val = math.log(val)
    if trans['scale']:
        val /= trans['scale']
    return val

def get_transforms(data):
    data_specs = data['dim_ranges']
    length_scales = data["guassian_params"]["length_scales"]

    transforms = dict()
    for name in length_scales:
        specs =  data_specs[name]
        if specs['type'] == 'continuous':
            start,end = specs['range']
            scale_val = (end-start)*length_scales[name]
            transforms[name] = {
                "scale":scale_val,
                "log": False
            }
        elif specs['type'] == 'multiplicative':
            start,end = specs['range']
            startl = math.log(start)
            endl = math.log(end)
            scale_val = (endl-startl)*length_scales[name]
            transforms[name] = {
                "scale":scale_val,
                "log": True
            }
    return transforms

def get_ordered_bounds(data):
    trans = get_transforms(data)
    data_specs = data['dim_ranges']
    scaled_ranges = dict()
    for name,val in data_specs.items():
        if 'range' in val:
            t = trans[name]
            start,end = val['range']
            scaled_ranges[name] = [transform(t,start),transform(t,end)]
    return order_vals(data,scaled_ranges)

def get_vals_for_disc(data,discrete_label):
    data_specs = data['dim_ranges']
    ordering = data['contin_ordering']
    trans = get_transforms(data)

    ys = []
    xs = []
    for point in data['data_points']:
        disc_label = format_combs([format_disc_comb(name,val) for name,val in point['data'].items() if data_specs[name]['type'] == "discrete" ])
        if discrete_label == disc_label:
            cur_y = point['reward']
            cur_xs = {name:transform(trans[name],val) for name,val in point['data'].items() if name in ordering}
            cur_xs = order_vals(data,cur_xs)

            ys.append(cur_y)
            xs.append(cur_xs)

    #npy_xs = [np.zeros([0]) for ]
    if len(ys) == 0:
        return np.zeros([0]) ,np.zeros([0,len(ordering)])
    else:
        xnpy = np.stack(xs)
        ynpy = np.stack(ys)
        return xnpy,ynpy

def parse_data(data):
    DISC_SCALE = 100
    data_specs =  data['dim_ranges']

    discrete_vals = get_discrete_vals(data)

    all_combs = all_combinations(discrete_vals)
    vals = {l:get_vals_for_disc(data,l) for l in all_combs}
    return vals

def next_point(data,xs,ys,bounds):
    if len(xs) == 0:
        first_point = (bounds[:,1]-bounds[:,0])/2 + bounds[:,0]
        return first_point

    regressor = sklearn.gaussian_process.GaussianProcessRegressor(
        kernel=sklearn.gaussian_process.kernels.RBF(length_scale=1.0),# length scales get handled in data preprocessing
        alpha=data['guassian_params']['noise'],
        optimizer=None
    )
    t = len(xs)
    d = len(xs[0])
    beta = 0.2*math.log(t+1)*d #standar formula for beta
    regressor.fit(xs,ys)
    print(beta)
    def upper_confidence_bound(x):
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = np.stack([x])
        mean,stdev = regressor.predict(x, return_std=True)
        return mean + stdev * beta
    print(bounds)
    #print("bounds")
    #print(upper_confidence_bound(np.asarray([[0.00617284, 0.48765432]])))
    min_val = scipydirect.minimize(upper_confidence_bound,bounds)
    xval = min_val.x
    yval = upper_confidence_bound([xval])
    return xval,yval

if __name__ == "__main__":
    assert len(sys.argv) == 2 , "needs one parameter, the data filename."
    data = json.load(open(sys.argv[1]))

    vals = parse_data(data)
    print(vals)
    xs,ys = vals['count_0']
    bounds = get_ordered_bounds(data)
    res = next_point(data,xs,ys,bounds)
    print(res)
