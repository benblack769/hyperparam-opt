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

def parse_data(data,trans):
    xs = []
    ys = []
    for x in data['data_points']:
        xs.append(trans.transform_point(x['data']))
        ys.append(x['reward'])

    bounds = trans.get_bounds()
    if len(xs) == 0:
        return np.zeros([0]),np.zeros([0,len(data['dim_ranges'])]),bounds
    else:
        return np.asarray(ys),np.stack(xs),bounds

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
    def neg_upper_confidence_bound(x):
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = np.stack([x])
        mean,stdev = regressor.predict(x, return_std=True)
        return -(mean + stdev * beta)
    print(bounds)
    #print("bounds")
    #print(upper_confidence_bound(np.asarray([[0.00617284, 0.48765432]])))
    min_val = scipydirect.minimize(neg_upper_confidence_bound,bounds)
    xval = min_val.x
    yval = -neg_upper_confidence_bound([xval])
    return xval,yval

if __name__ == "__main__":
    assert len(sys.argv) == 2 , "needs one parameter, the data filename."
    data = json.load(open(sys.argv[1]))

    trans = Transformer(data)
    ys,xs,bounds = parse_data(data,trans)
    print(xs)
    print(ys)
    print(bounds)
    res = next_point(data,xs,ys,bounds)
    print(res)
    inv_res = trans.inverse_point(res[0])
    print(inv_res)
