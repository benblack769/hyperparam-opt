#import sklearn
import sys
import json
import numpy as np
import math
import sklearn
import sklearn.gaussian_process
import scipydirect
from transformer import Transformer

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
