#import sklearn
import sys
import json
import numpy as np
import math
import sklearn
import sklearn.gaussian_process
import scipydirect

class Transformer:
    def __init__(self,data):
        self.data_specs = data['dim_ranges']
        #self.length_scales = data["guassian_params"]["length_scales"]
        self.data = data

        self.ordering = [name for name in self.data_specs]
        self.transforms = self.order(self.get_transforms(data))

    def order(self,vals):
        ordered_scales = []
        for name in self.ordering:
            ordered_scales.append(vals[name])
        return ordered_scales

    def transform_point(self,val):
        result = []
        for name,trans in zip(self.ordering,self.transforms):
            cur_val = val[name]
            if trans['disc_val']:
                for x in range(trans['disc_val']):
                    lval = trans['scale'] if x == cur_val else 0
                    result.append(lval)
            else:
                if trans['log']:
                    cur_val = math.log(cur_val)
                if trans['scale']:
                    cur_val /= trans['scale']
                result.append(cur_val)

        return np.asarray(result,dtype=np.float64)

    def get_bounds(self):
        result = []
        for name,trans in zip(self.ordering,self.transforms):
            if trans['disc_val']:
                for x in range(trans['disc_val']):
                    result.append(np.asarray([0,trans['scale']],dtype=np.float64))
            else:
                currange = np.asarray(self.data_specs[name]['range'],dtype=np.float64)
                if trans['log']:
                    currange = np.log(currange)
                if trans['scale']:
                    currange /= trans['scale']
                result.append(currange)

        return np.stack(result)

    def inverse_point(self,np_arr):
        cur_idx = 0
        res = dict()
        for name,trans in zip(self.ordering,self.transforms):
            if trans['disc_val']:
                count = trans['disc_val']
                indexed = [(np_arr[x+cur_idx],x) for x in range(count)]
                best_idx = min(indexed)[1]
                res[name] = best_idx
                cur_idx += count
            else:
                cur_val = np_arr[cur_idx]
                if trans['scale']:
                    cur_val *= trans['scale']
                if trans['log']:
                    cur_val = math.exp(cur_val)
                res[name] = cur_val
                cur_idx += 1

        assert cur_idx == len(self.ordering)
        return res

    def get_transforms(self,data):
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
                    "log": False,
                    "disc_val": 0
                }
            elif specs['type'] == 'multiplicative':
                start,end = specs['range']
                startl = math.log(start)
                endl = math.log(end)
                scale_val = (endl-startl)*length_scales[name]
                transforms[name] = {
                    "scale":scale_val,
                    "log": True,
                    "disc_val": 0
                }
            elif specs['type'] == 'discrete':
                scale_val = length_scales[name]
                transforms[name] = {
                    "scale":scale_val,
                    "log": False,
                    "disc_val": specs['possibilities']
                }

        return transforms

def parse_data(data,trans,fidelity):
    xs = []
    ys = []
    for x in data['data_points']:
        if x['fidelity'] == fidelity:
            xs.append(trans.transform_point(x['data']))
            ys.append(x['reward'])

    if len(xs) == 0:
        return np.zeros([0]),np.zeros([0,len(data['dim_ranges'])])
    else:
        return np.asarray(ys),np.stack(xs)

def next_point(data,trans):
    num_fidelities = data['num_fidelities']
    multi_fid_values = data['multi_fidelity_params']

    bounds = trans.get_bounds()
    if len(data['data_points']) == 0:
        first_point = (bounds[:,1]-bounds[:,0])/2 + bounds[:,0]
        first_fidelity = 0
        return first_point,first_fidelity

    regressors = []
    for fidelity in range(num_fidelities):
        reg = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=sklearn.gaussian_process.kernels.RBF(length_scale=1.0),# length scales get handled in data preprocessing
            alpha=data['guassian_params']['noise'],
            optimizer=None
        )
        ys,xs = parse_data(data,trans,fidelity)
        if len(ys) > 0:
            reg.fit(xs,ys)

        regressors.append(reg)

    t = len(data['data_points'])
    d = len(data['data_points'][0]['data'])
    beta = 0.2*math.log(t+1)*d #standard formula for beta
    zetas = multi_fid_values['err_bounds'] + [0.0]
    def neg_upper_confidence_bound(x):
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = np.stack([x])
        ucbs = []
        for zeta,reg in zip(zetas,regressors):
            mean,stdev = reg.predict(x, return_std=True)
            ucbs.append(mean + stdev * beta + zeta)
        true_ucb = min(ucbs)
        return -true_ucb
    print(bounds)
    #print("bounds")
    #print(upper_confidence_bound(np.asarray([[0.00617284, 0.48765432]])))
    min_val = scipydirect.minimize(neg_upper_confidence_bound,bounds)
    xval = min_val.x

    acc_targets = multi_fid_values['accuracy_targets']+[0.0]
    out_fid_level = num_fidelities-1# defaults to highest fidelity function
    for fid_level,(acc,reg) in enumerate(zip(acc_targets,regressors)):
        mean,stdev = reg.predict([min_val.x], return_std=True)
        if stdev*beta > acc:
            out_fid_level = fid_level
            break

    yval = -neg_upper_confidence_bound([xval])
    return xval,yval,out_fid_level

if __name__ == "__main__":
    assert len(sys.argv) == 2 , "needs one parameter, the data filename."
    data = json.load(open(sys.argv[1]))

    trans = Transformer(data)
    #ys,xs = parse_data(data,trans)
    #bounds = trans.get_bounds()
    #print(xs)
    #print(ys)
    #print(bounds)
    res = next_point(data,trans)
    print(res)
    inv_res = trans.inverse_point(res[0])
    print(inv_res)
