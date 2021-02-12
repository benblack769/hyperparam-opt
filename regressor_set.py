import numpy as np
from transformer import Transformer
import time
import random
from mydirect import direct
import math
import json
from neural_net_regressor import NNRegressor

DOT_SIG_VAL = 0.1

class DataPoint:
    def __init__(self,coord,reward,group,fid_level,stdev):
        self.coord = coord
        self.reward = reward
        self.group = group
        self.fid_level = fid_level
        self.err_on_select = float(stdev)

    def to_dict(self):
        return {
            "coord":(self.coord),
            "reward":self.reward,
            "group":self.group,
            "fid_level":self.fid_level,
            "err_on_select":self.err_on_select,
        }
    def to_temp(self):
        return TempPoint(
            self.coord,
            self.reward,
            self.group,
            self.fid_level,
            self.err_on_select,
        )

    def to_data(self,reward):
        return DataPoint(
            self.coord,
            reward,
            self.group,
            self.fid_level,
            self.err_on_select,
        )

class TempPoint(DataPoint):
    def __init__(self,*args):
        super().__init__(*args)
        self.timestamp = time.time()
    def expired(self,limit):
        return time.time() - self.timestamp > limit


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


        self.MAX_LEVEL = config.get('max_scale_val',0.5)
        self.MEDIAN_LEVEL = config.get('median_scale_val',-1.0)
        self.num_nns = config.get('num_nns', 100)
        self.nn_hidden_size = config.get('nn_hidden_size', 256)
        self.device = config.get('device', 'cpu')

        self.noise_hyperparam = 0.1
        ORIG_LEN_SCALE = 1
        self.length_scales = {name: ORIG_LEN_SCALE for name in config['dim_ranges']}

        aprox_stdev = config['aprox_reward_stdev']
        # accuracy_bounds
        self.accuracy_bounds = [aprox_stdev/10]*(num_fidelities-1)+[0.0]
        # accuracy_targets
        self.accuracy_targets = [aprox_stdev/10]*(num_fidelities-1)+[0.0]
        #x eD different regressors for the different fidelity levels
        self.regressors = [None]*num_fidelities
        self.transformer = Transformer(config['dim_ranges'],self.length_scales)
        self.cur_mean = 0.0
        self.cur_max = 0.0

        for fid_level in range(self.num_fidelities):
            self.regressors[fid_level] = NNRegressor(self.num_nns, self.nn_hidden_size, device=self.device)
            self.retrain(fid_level)
        num_trans_dims = self.transformer.num_transformed_dims()
        for x in range(2*len(self.length_scales)):
            invpoint = np.random.uniform(0,1/ORIG_LEN_SCALE,size=num_trans_dims)
            point = self.transformer.inverse_point(invpoint)
            self.queued_points.append(DataPoint(point,0.0,self.current_group,0,0.01))
            self.current_group += 1

    def remove_old_temp_points(self):
        time_limit = self.config['timeout_evaluation']
        self.temp_points = [point for point in self.temp_points
                                    if not point.expired(time_limit)]

    def rewards(self):
        all_points = self.data_points+self.temp_points
        rewards = [x.reward for x in all_points]
        return np.asarray(rewards,dtype=np.float64)

    def retrain(self,fid_level):
        #if self.autotune and len(self.data_points) % 10 == 6:
        #    autotuned_res = autotune(self.config,self.data_points)
        #print("lenpoints: ", len(self.data_points))
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

        self.cur_mean = median_reward
        self.cur_max = max_reward

        ys = self.transform_y(ys)

        #print(ys)
        #print(xs)
        self.regressors[fid_level].fit(xs,ys)


    def transform_y(self,ys):
        ys = ys - self.cur_mean
        ys /= (self.cur_max - self.cur_mean + 1e-10)
        ys *= (self.MAX_LEVEL - self.MEDIAN_LEVEL + 1e-10)
        ys -= self.MEDIAN_LEVEL
        return ys

    def inverse_y(self,ys):
        ys = ys + self.MEDIAN_LEVEL
        ys /= (self.MAX_LEVEL - self.MEDIAN_LEVEL + 1e-10)
        ys *= (self.cur_max - self.cur_mean + 1e-10)
        ys += self.cur_mean
        return ys

    def transform_stdev(self,stdev):
        stdev = stdev * (self.MAX_LEVEL - self.MEDIAN_LEVEL + 1e-10)
        stdev /= (self.cur_max - self.cur_mean + 1e-10)
        return stdev

    def inverse_stdev(self,stdev):
        stdev = stdev / (self.MAX_LEVEL - self.MEDIAN_LEVEL + 1e-10)
        stdev *= (self.cur_max - self.cur_mean + 1e-10)
        return stdev

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
            "accuracy_bounds": self.accuracy_bounds,
            "accuracy_targets": self.accuracy_targets,
        })

    def process_accuracy_bounds(self,group):
        group_points = [point for point in self.data_points if point.group == group]
        group_points.sort(key=lambda x: x.fid_level)

        for low_idx in range(len(group_points)):
            for high_idx in range(low_idx+1,len(group_points)):
                low_fid_p = group_points[low_idx]
                high_fid_p = group_points[high_idx]
                low_acc = self.accuracy_bounds[low_fid_p.fid_level]
                high_acc = self.accuracy_bounds[high_fid_p.fid_level]

                if low_acc - high_acc < abs(low_fid_p.reward - high_fid_p.reward):
                    self.accuracy_bounds[low_fid_p.fid_level] = high_acc + 1.2*abs(low_fid_p.reward - high_fid_p.reward)

    def process_accuracy_targets(self):
        def get_acc_targ(window,fid,k_stat):
            if len(all_points) < window+1:
                return 1e-50
            start_window = len(all_points)-window
            prev_fid_points = [point.err_on_select for point in all_points[start_window:] if point.fid_level == fid]
            prev_stev_vals = np.asarray(prev_fid_points)

            ordered = np.sort(prev_stev_vals)
            #print(ordered)
            if len(ordered)+1 < k_stat:
                return 1e-50
            else:
                return ordered[k_stat-1]

        fid_marginal_costs = self.config['aprox_fid_costs']
        fid_total_costs = [0]*self.num_fidelities
        all_points = self.data_points+self.temp_points
        #DECAY_PARAM = 0.99
        for point in all_points:
            fid_total_costs[point.fid_level] += fid_marginal_costs[point.fid_level]
        print("tot_costs: ",fid_total_costs)

        for i in range(1,self.num_fidelities):
            cost_ratio = fid_marginal_costs[i]/fid_marginal_costs[0]
            window_size = 3
            window_five = int(cost_ratio*window_size)
            #count_one = count_onewindow(window_one,i)
            targ = get_acc_targ(window_five,i-1,window_size)
            self.accuracy_targets[i-1] = targ


        print("acctargs: ",self.accuracy_targets)
        print("transacctargs: ",self.transform_stdev(np.array(self.accuracy_targets)))

    def _calc_next_points(self,debug=True):
        num_fidelities = self.num_fidelities

        bounds = self.transformer.get_bounds()
        t = len(self.data_points) + len(self.temp_points)
        if t == 0:
            first_point_reg = (bounds[:,1]-bounds[:,0])/2 + bounds[:,0]
            first_point = self.transformer.inverse_point(first_point_reg)
            first_fidelity = 0
            return first_point,first_fidelity,0

        d = len(self.config['dim_ranges'])
        beta = 0.2*math.log(t+1)*d #standard formula for beta
        zetas = self.transform_stdev(np.array(self.accuracy_bounds))

        if debug:
            print("zetas: ",zetas)
        def neg_upper_confidence_bound(x):
            x = np.asarray(x)
            if len(x.shape) == 1:
                x = np.stack([x])
            ucbs = []
            for zeta,reg in zip(zetas,self.regressors):
                mean,stdev = reg.predict(x, return_std=True)
                ucbs.append(mean + stdev * beta + zeta)
            ucbs = np.stack(ucbs)
            true_ucb = np.min(ucbs,axis=0)
            return -true_ucb
        min_y,min_x = direct(neg_upper_confidence_bound,bounds,maxsample=2000)

        acc_targets = self.accuracy_targets
        acc_targets = self.transform_stdev(np.array(acc_targets))

        out_fid_level = num_fidelities-1# defaults to highest fidelity function
        for fid_level,(acc,reg) in enumerate(zip(acc_targets,self.regressors)):
            mean,stdev = reg.predict([min_x], return_std=True)
            stdev = stdev[0]
            if fid_level == 0:
                zero_stdev = stdev/self.noise_hyperparam
            if stdev > acc:
                out_fid_level = fid_level
                out_stdevs = stdev/self.noise_hyperparam
                break

        xval = self.transformer.inverse_point(min_x)
        #yval = -neg_upper_confidence_bound([xval])
        return xval,out_fid_level,out_stdevs,zero_stdev

    def _queue_point(self):
        out_point,out_fidlevel,out_stdevs,_ = self._calc_next_points()

        print("noise level")
        print(self.noise_hyperparam)
        print("out_stevs")
        print(out_stdevs)

        group_num = self.current_group
        self.current_group += 1
        tranformed_point = self.transformer.transform_point(out_point)
        for fid_level in range(out_fidlevel+1):
            reg = self.regressors[fid_level]
            prior_val,prior_std = reg.predict([tranformed_point],return_std=True)
            prior_val = self.inverse_y(float(prior_val))
            prior_std = self.inverse_stdev(float(prior_std))
            self.queued_points.append(DataPoint(out_point,prior_val,group_num,fid_level,prior_std))

    def generate_next_point(self):
        if not len(self.queued_points):
            self._queue_point()

        out_point = self.queued_points[0]
        self.queued_points.pop(0)
        self.temp_points.append(out_point.to_temp())
        self.retrain(out_point.fid_level)
        self.process_accuracy_targets()
        print("accbounds",self.accuracy_bounds)

        return out_point

    def add_point(self,point):
        for i,p in enumerate(self.temp_points):
            if p.group == point.group and p.fid_level == point.fid_level:
                self.temp_points.pop(i)
                break
        self.data_points.append(point)

        self.process_accuracy_bounds(point.group)
        self.retrain(point.fid_level)
