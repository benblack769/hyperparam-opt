import numpy as np
from transformer import Transformer
import time
import sklearn
import random
import sklearn.gaussian_process
import scipydirect
import math
import json

class DataPoint:
    def __init__(self,coord,reward,group,fid_level):
        self.coord = coord
        self.reward = reward
        self.group = group
        self.fid_level = fid_level

    def to_dict(self):
        return {
            "coord":(self.coord),
            "reward":self.reward,
            "group":self.group,
            "fid_level":self.fid_level,
        }
    def to_temp(self):
        return TempPoint(
            self.coord,
            self.reward,
            self.group,
            self.fid_level
        )

class TempPoint(DataPoint):
    def __init__(self,*args):
        super().__init__(*args)
        self.timestamp = time.time()
    def expired(self,limit):
        return time.time() - self.timestamp > limit

def autotune_metric(coord,config,data_points):
    assert len(data_points) > 3,"needs more points to properly evaluate hyperparams"
    #data_points = list(data_points)
    #random.shuffle(data_points)
    noise_level = coord['__noise__']
    #length_scales = {name:c for name,c in coord.items() if name != "__noise__"}
    dsize = len(data_points)
    twofold_split = [
        data_points[:dsize//2],
        data_points[dsize//2:],
    ]
    regset = RegressorSet(config)
    #regset.length_scales = length_scales
    regset.noise_hyperparam = noise_level
    tot = 0
    for split in range(2):
        train_data = twofold_split[0^split]
        test_data = twofold_split[1^split]
        regset.data_points = train_data
        regset.retrain(0)
        xs = np.stack([regset.transformer.transform_point(x.coord) for x in test_data])
        ys = np.asarray([x.reward for x in test_data],dtype=np.float64)
        predys,stdevs = regset.regressors[0].predict(xs,return_std=True)
        num_stdevs = np.abs(ys-predys)**3/stdevs**2
        tot += np.sum(num_stdevs)
    return tot


def autotune_config(config):
    # dim_ranges = {name: {
    #     "type": "multiplicative",
    #     "range": [1e-2,1.0]
    # } for name in config['dim_ranges']}
    dim_ranges = dict()
    dim_ranges["__noise__"] = {
        "type": "multiplicative",
        "range": [1e-7,1e2]
    }

    new_config = {
        "num_fidelities": 1,
        "aprox_reward_stdev": 0.5,
        "aprox_fid_costs": [1],
        "timeout_evaluation": 1000,
        "autotune": False,
        "dim_ranges": dim_ranges,
    }
    return new_config

def binary_search(func,start,end,depth_to_end):
    if depth_to_end <= 0:
        return start
    qval = (end - start) / 4
    low = start + qval
    high = start + qval * 3
    vallow = func(low)
    valhigh = func(high)
    mid = start + qval * 2
    if vallow > valhigh:
        return binary_search(func,start,mid,depth_to_end-1)
    else:
        return binary_search(func,mid,end,depth_to_end-1)


def autotune(config,data_points):
    regset = RegressorSet(autotune_config(config))
    best_point = None
    for x in range(100):
        exec_point = regset.generate_next_point()
        out_value = autotune_metric(exec_point.coord,config,data_points)
        new_point = DataPoint(exec_point.coord,out_value,exec_point.group,exec_point.fid_level)
        regset.add_point(new_point)
        print(json.dumps(new_point.to_dict(),indent=2))
        if not best_point or new_point.reward > best_point.reward:
            best_point = new_point

    coord = best_point.coord
    noise_level = coord['__noise__']
    #length_scales = {name:c for name,c in coord.items() if name != "__noise__"}

    return noise_level


class RegressorSet:
    def __init__(self,config):
        self.num_fidelities = num_fidelities = config['num_fidelities']
        # temp points have a timestamp of when they were sent out,
        # explire when time since request exceeds config value `timeout_evaluation`
        self.temp_points = []
        # queued points are those that are determined by the algorithm but
        # not yet sent out by generate_next_point
        self.queued_points = []
        # data points are the actual data that are calculated with this method
        # have three values:
        # fidelity,reward,point,group
        self.data_points = []
        # a group is a set of  evaluations at different fidelities
        # but at the same point. Used for heuristic calculation.
        self.current_group = 0
        self.config = config
        self.autotune = config['autotune']

        self.noise_hyperparam = 0.1
        self.length_scales = {name: 0.5 for name in config['dim_ranges']}

        aprox_stdev = config['aprox_reward_stdev']
        # accuracy_bounds
        self.accuracy_bounds = [aprox_stdev/10]*(num_fidelities-1)+[0.0]
        # accuracy_targets
        self.accuracy_targets = [aprox_stdev/10]*(num_fidelities-1)+[0.0]
        # different regressors for the different fidelity levels
        self.regressors = [None]*num_fidelities
        self.transformer = Transformer(config['dim_ranges'],self.length_scales)

        self.reset_hyperparams()

    def reset_hyperparams(self):
        self.transformer = Transformer(self.config['dim_ranges'],self.length_scales)

        for fid_level in range(self.num_fidelities):
            self.retrain(fid_level)

    def remove_old_temp_points(self):
        time_limit = self.config['timeout_evaluation']
        self.temp_points = [point for point in self.temp_points
                                    if not point.expired(time_limit)]

    def rewards(self):
        all_points = self.data_points+self.temp_points
        rewards = [x.reward for x in all_points]
        return np.asarray(rewards,dtype=np.float64)

    def retrain(self,fid_level):
        self.regressors[fid_level] = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=sklearn.gaussian_process.kernels.RBF(length_scale=1.0),# length scales get handled in data preprocessing
            alpha=self.noise_hyperparam,
            optimizer=None,
            #normalize_y=True
        )
        #if self.autotune and len(self.data_points) % 10 == 6:
        #    autotuned_res = autotune(self.config,self.data_points)
        #    print(autotuned_res)

        self.remove_old_temp_points()
        all_points = self.data_points+self.temp_points
        fid_points = [point for point in all_points if point.fid_level == fid_level]
        if not fid_points:
            return
        xs = np.stack([self.transformer.transform_point(p.coord) for p in fid_points])
        ys = np.asarray([p.reward for p in fid_points],dtype=np.float64)
        all_rewards = self.rewards()
        max_reward = np.max(all_rewards)
        median_reward = np.median(all_rewards)
        ys -= median_reward
        ys /= 2*(max_reward-median_reward+1e-10)
        print(ys)
        #print(ys)
        self.regressors[fid_level].fit(xs,ys)

    def load_data(self,data):
        pass

    def extract_data(self):
        return json.dumps({
            "num_fidelities": self.num_fidelities,
            "temp_points": [p.to_dict() for p in self.temp_points],
            "queued_points": [p.to_dict() for p in self.queued_points],
            "data_points": [p.to_dict() for p in self.data_points],
            "current_group": self.current_group,
            "config": self.config,
            "noise_hyperparam": self.noise_hyperparam,
            "length_scales": self.length_scales,
            "accuracy_bounds": self.accuracy_bounds,
            "accuracy_targets": self.accuracy_targets,
        })

    def process_accuracy_bounds(self,group):
        group_points = [point for point in self.data_points if point.group == group]
        group_points.sort(key=lambda x: x.fid_level)

        for low_idx in range(group_points):
            for high_idx in range(i+1,group_points):
                low_fid_p = group_points[low_idx]
                high_fid_p = group_points[high_idx]
                low_acc = self.accuracy_bounds[low_fid_p.fid_level]
                high_acc = self.accuracy_bounds[high_fid_p.fid_level]

                if low_acc - high_acc > abs(low_fid_p.reward - high_fid_p.reward):
                    self.accuracy_bounds[low_acc] *= 1.1

    def process_accuracy_targets(self):
        fid_marginal_costs = self.config['aprox_fid_costs']
        fid_total_costs = [0]*self.num_fidelities
        for point in (self.data_points+self.temp_points):
            fid_total_costs[point.fid_level] += fid_marginal_costs[point.fid_level]

        for i in range(len(fid_total_costs)-1):
            if fid_total_costs[i] > fid_total_costs[i+1]+fid_marginal_costs[i+1]*10:
                self.accuracy_targets[i] *= 1.1

    def _calc_next_points(self):
        num_fidelities = self.num_fidelities

        bounds = self.transformer.get_bounds()
        t = len(self.data_points) + len(self.temp_points)
        if t == 0:
            first_point_reg = (bounds[:,1]-bounds[:,0])/2 + bounds[:,0]
            first_point = self.transformer.inverse_point(first_point_reg)
            first_fidelity = 0
            return first_point,first_fidelity

        d = len(self.config['dim_ranges'])
        beta = 0.2*math.log(t+1)*d #standard formula for beta
        zetas = self.accuracy_bounds

        def neg_upper_confidence_bound(x):
            x = np.asarray(x)
            if len(x.shape) == 1:
                x = np.stack([x])
            ucbs = []
            for zeta,reg in zip(zetas,self.regressors):
                mean,stdev = reg.predict(x, return_std=True)
                ucbs.append(mean + stdev * beta + zeta)
            true_ucb = min(ucbs)
            return -true_ucb
        #print("bounds")
        #print(upper_confidence_bound(np.asarray([[0.00617284, 0.48765432]])))
        min_val = scipydirect.minimize(neg_upper_confidence_bound,bounds,algmethod=1)

        acc_targets = self.accuracy_targets
        out_fid_level = num_fidelities-1# defaults to highest fidelity function
        for fid_level,(acc,reg) in enumerate(zip(acc_targets,self.regressors)):
            mean,stdev = reg.predict([min_val.x], return_std=True)
            if stdev*beta > acc:
                out_fid_level = fid_level
                break

        xval = self.transformer.inverse_point(min_val.x)
        #yval = -neg_upper_confidence_bound([xval])
        return xval,out_fid_level

    def _queue_point(self):
        out_point,out_fidlevel = self._calc_next_points()
        group_num = self.current_group
        self.current_group += 1
        tranformed_point = self.transformer.transform_point(out_point)
        for fid_level in range(out_fidlevel+1):
            reg = self.regressors[fid_level]
            prior_val = reg.predict([tranformed_point])
            self.queued_points.append(DataPoint(out_point,prior_val,group_num,fid_level))

    def generate_next_point(self):
        if not len(self.queued_points):
            self._queue_point()

        out_point = self.queued_points[0]
        self.queued_points.pop(0)
        self.temp_points.append(out_point.to_temp())
        self.retrain(out_point.fid_level)
        return out_point

    def add_point(self,point):
        for i,p in enumerate(self.temp_points):
            if p.group == point.group and p.fid_level == point.fid_level:
                self.temp_points.pop(i)
                break
        self.data_points.append(point)
        self.retrain(point.fid_level)
