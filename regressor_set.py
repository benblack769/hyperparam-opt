import numpy as np
from transformer import Transformer
import time
import sklearn
import random
import sklearn.gaussian_process
from mydirect import direct
import math
import json

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

def autotune_metric(noise_level,config,data_points,length_scales):
    regset = RegressorSet(config)
    regset.data_points = data_points
    regset.transformer = Transformer(config['dim_ranges'],length_scales)
    out_noise = -1e100
    fid_level = 0
    kernel = (sklearn.gaussian_process.kernels.RBF(length_scale=1.0)
            #+ sklearn.gaussian_process.kernels.DotProduct(sigma_0=DOT_SIG_VAL)
    )
    regset.regressors[fid_level] = sklearn.gaussian_process.GaussianProcessRegressor(
        kernel=kernel,# length scales get handled in data preprocessing
        alpha=noise_level,
        optimizer=None
    )
    regset.retrain(fid_level)
    new_value = regset.regressors[fid_level].log_marginal_likelihood_value_
    #cur_noise = regset.regressors[fid_level].kernel_
    #out_noise = max(out_noise,regset.regressors[fid_level].alpha_)
    #out_noise = newkern.k2.noise_level
    return new_value


def autotune_direct(config,data_points,length_scales):
    def eval_fn(x):
        val = math.exp(x)
        out_value = autotune_metric(val,config,data_points,length_scales)
        return -out_value

    def batch_evan_fn(xs):
        return np.stack([eval_fn(x) for x in xs])

    search_samples = 80
    bounds = [[math.log(1e-7),math.log(1e2)]]
    _,out_noise_log = direct(batch_evan_fn,bounds,search_samples)
    out_noise = math.exp(out_noise_log)

    return out_noise


def autotune_sklearn(config,data_points,length_scales):
    if len(data_points) < 2:
        return 0.1
    regset = RegressorSet(config)
    regset.data_points = data_points
    regset.transformer = Transformer(config['dim_ranges'],length_scales)
    out_noise = -1e100
    fid_level = 0
    kernel = (sklearn.gaussian_process.kernels.RBF(length_scale=1.0, length_scale_bounds=(1.0, 1.0))
        #+ sklearn.gaussian_process.kernels.DotProduct(sigma_0=DOT_SIG_VAL, sigma_0_bounds=(DOT_SIG_VAL,DOT_SIG_VAL))
        + sklearn.gaussian_process.kernels.WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-5, 1e4))
    )
    regset.regressors[fid_level] = sklearn.gaussian_process.GaussianProcessRegressor(
        kernel=kernel,# length scales get handled in data preprocessing
        alpha=0.0,
        n_restarts_optimizer=10
    )
    regset.retrain(fid_level)
    newkern = regset.regressors[fid_level].kernel_
    #cur_noise = regset.regressors[fid_level].kernel_
    #out_noise = max(out_noise,regset.regressors[fid_level].alpha_)
    out_noise = newkern.k2.noise_level
    return out_noise

def set_value(config,name,default):
    return default if name not in config else config[name]

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
        self.low_noise_sel_count = 0
        self.config = config
        self.autotune = config['autotune']

        self.INTOLERABLE_NOISE_LEVEL = set_value(config,'intolerable_noise_level',1.1)# if in
        self.MAX_LEVEL = set_value(config,'max_scale_val',0.5)
        self.MEDIAN_LEVEL = set_value(config,'median_scale_val',-1.0)

        self.noise_hyperparam = 0.1
        ORIG_LEN_SCALE = 10*math.sqrt(len(config['dim_ranges']))
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

        self.reset_hyperparams()
        num_trans_dims = self.transformer.num_transformed_dims()
        for x in range(2*len(self.length_scales)):
            invpoint = np.random.uniform(0,1/ORIG_LEN_SCALE,size=num_trans_dims)
            point = self.transformer.inverse_point(invpoint)
            self.queued_points.append(DataPoint(point,0.0,self.current_group,0,0.01))
            self.current_group += 1

    def reset_hyperparams(self):
        self.transformer = Transformer(self.config['dim_ranges'],self.length_scales)

        if self.data_points:
            self.noise_hyperparam = autotune_direct(self.config,self.data_points,self.length_scales)

        for fid_level in range(self.num_fidelities):
            kernel = (
                sklearn.gaussian_process.kernels.RBF(length_scale=1.0)  #, length_scale_bounds=(1.0, 1.0))# \
                #+ sklearn.gaussian_process.kernels.DotProduct(sigma_0=DOT_SIG_VAL)
            )
            #    + sklearn.gaussian_process.kernels.WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+5))

            self.regressors[fid_level] = sklearn.gaussian_process.GaussianProcessRegressor(
                kernel=kernel,# length scales get handled in data preprocessing
                alpha=self.noise_hyperparam,
                optimizer=None,
                #normalize_y=True
            )
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
            "length_scales": self.length_scales,
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

    def decrese_length_scale(self):
        SCALE_DEC = 0.6
        min_scale_val = -1e100
        min_scale_name = ""
        print("scale vals:")
        for name in self.length_scales.keys():
            self.length_scales[name] *= SCALE_DEC
            self.reset_hyperparams()
            _,_,_,cur_scale_val = self._calc_next_points(False)
            #cur_noise = self.noise_hyperparam
            print(cur_scale_val)
            self.length_scales[name] /= SCALE_DEC
            #print(name,cur_noise)
            if cur_scale_val > min_scale_val:
                min_scale_val = cur_scale_val
                min_scale_name = name

        largest_scale = max(self.length_scales.values())
        if self.length_scales[min_scale_name]*10 > largest_scale:
            self.length_scales[min_scale_name] *= SCALE_DEC
        else:
            for name in self.length_scales:
                self.length_scales[name] *= SCALE_DEC
        print(self.length_scales.values())
        self.reset_hyperparams()


    def _queue_point(self):
        out_point,out_fidlevel,out_stdevs,_ = self._calc_next_points()

        print("noise level")
        print(self.noise_hyperparam)
        print("out_stevs")
        print(out_stdevs)
        if out_stdevs < self.INTOLERABLE_NOISE_LEVEL:
            self.low_noise_sel_count += 1
        if self.low_noise_sel_count >= 1:
            self.low_noise_sel_count = 0
            while out_stdevs < self.INTOLERABLE_NOISE_LEVEL:
                self.decrese_length_scale()
                print("noise adjusted")
                print(self.noise_hyperparam)
                out_point,out_fidlevel,out_stdevs,_ = self._calc_next_points()


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

        self.reset_hyperparams()
        self.process_accuracy_bounds(point.group)
        #self.retrain(point.fid_level)
