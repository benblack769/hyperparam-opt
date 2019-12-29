import numpy as np
import math

class Transformer:
    def __init__(self,data_specs,length_scales):
        self.data_specs = data_specs
        self.length_scales = length_scales

        self.ordering = [name for name in self.data_specs]
        unordered_transforms = self.get_transforms(data_specs,length_scales)
        self.transforms = self.order(unordered_transforms)

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

    def get_transforms(self,data_specs,length_scales):
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
