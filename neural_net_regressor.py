import torch
from torch import nn
import numpy as np
import math

class BatchedLinear(nn.Module):
    def __init__(self, num_nns, in_size, out_size):
        super().__init__()
        scale = 1/max(1., (in_size+out_size)/2.)
        limit = math.sqrt(3.0 * scale)
        self.weight = torch.nn.Parameter(torch.zeros(num_nns, out_size, in_size), requires_grad=True)
        self.weight.data = torch.rand((num_nns, out_size, in_size))*limit*2-limit
        self.bias = torch.nn.Parameter(torch.zeros(num_nns, out_size, 1), requires_grad=True)

    def forward(self, obs):
        # print(obs.shape, self.weight.shape)
        # print(obs.shape, (torch.bmm(self.weight, obs)).shape)
        return (torch.bmm(self.weight, obs) + self.bias)

class NNModel(nn.Module):
    def __init__(self, num_nns, in_size, hidden_size):
        super().__init__()
        self.num_nns = num_nns
        self.model = nn.Sequential(
            BatchedLinear(num_nns, in_size, hidden_size),
            nn.ELU(),
            BatchedLinear(num_nns, hidden_size, hidden_size),
            nn.ELU(),
            BatchedLinear(num_nns, hidden_size, 1),
        )

    def forward(self, obs):
        obs = obs.unsqueeze(0).repeat(self.num_nns,1,1).transpose(1,2)
        out = self.model(obs)
        return out.reshape(self.num_nns, -1)

class NNRegressor:
    def __init__(self, num_nns, hidden_size, num_iters=250, device="cpu"):
        self.num_nns = num_nns
        self.hidden_size = hidden_size
        self.device = device
        self.num_iters = num_iters
        self.model = None

    def fit(self, Xs, ys):
        in_size = Xs.shape[1]
        Xs = torch.tensor(Xs,device=self.device).float()
        ys = torch.tensor(ys,device=self.device).float()
        self.model = NNModel(self.num_nns, in_size, self.hidden_size).to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(),lr=0.01)
        for i in range(self.num_iters):
            self.model.zero_grad()
            pred_outs = self.model(Xs)
            losses = torch.mean((pred_outs - ys)**2)
            losses.backward()
            optimizer.step()

    def predict(self, Xs, return_std=False):
        if self.model is None:
            mean_pred = np.zeros(Xs.shape[0])
            stddev_pred = np.ones(Xs.shape[0])
        else:
            Xs = torch.tensor(Xs,device=self.device).float()
            all_preds = self.model(Xs)
            mean_pred = torch.mean(all_preds,axis=0).detach().double().cpu().numpy()
            stddev_pred = torch.std(all_preds,axis=0).detach().double().cpu().numpy()
            print(all_preds)
            print(mean_pred)
        if return_std:
            return mean_pred, stddev_pred
        else:
            return mean_pred
