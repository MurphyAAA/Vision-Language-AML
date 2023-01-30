# -*- coding: utf-8 -*-
"""
@Time ： 2023/1/23 18:37
@Auth ： Murphy
@File ：my_loss.py
@IDE ：PyCharm
"""
import torch
import torch.nn as nn
from parse_args import parse_arguments
class DomainDisentangleLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super(DomainDisentangleLoss, self).__init__()
        self.nll_loss = torch.nn.NLLLoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.rec_loss = torch.nn.MSELoss()

        self.alpha1 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True) if weight is None else weight
        self.alpha2 = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # weights are hyperparameters
        self.w1 = 1
        self.w2 = 1
        self.w3 = 1
        opt = parse_arguments()
        self.batch_size = opt['batch_size']
    def forward(self, fG, fG_hat, Cfcs, DCfcs, DCfds, Cfds, y_s, yd): # 这个input本身就是model处理的结果所以不用将model 传入loss类了

        # c_ent_l1 = self.entropy_loss(DCfcs[0:self.batch_size, :])
        # c_ent_l2 = self.entropy_loss(DCfcs[self.batch_size: , :])
        # print(torch.log(Cfcs).size(),y_s.size())
        L_class = self.nll_loss(torch.log(Cfcs),y_s)# + self.alpha1*self.entropy_loss(DCfcs)
        # L_domain = self.cross_entropy_loss(DCfds,yd) + self.alpha2*self.entropy_loss(Cfds)
        # d_ent_l1 = self.entropy_loss(Cfds[0:self.batch_size, :])
        # d_ent_l2 = self.entropy_loss(Cfds[self.batch_size:, :])
        l1 = self.nll_loss(torch.log(DCfds[0:y_s.size(0),:]) , yd[0:y_s.size(0)])
        l2 = self.nll_loss(torch.log(DCfds[y_s.size(0):,:]) , yd[y_s.size(0):])
        L_domain = l1+l2# + self.alpha2*self.entropy_loss(Cfds)
        # L_domain = self.nll_loss(torch.log(DCfds),yd)# + self.alpha2*self.entropy_loss(Cfds)

        tot_loss = self.w1 * L_class + self.w2 * L_domain  + self.w3 * self.rec_loss(fG,fG_hat)
        return tot_loss

    def entropy_loss(self, f):  # 应该返回一个标量 最后是求和的
        #  ?Freeze?

        logf = torch.log(f)
        # logf1 = logf[0:logf.size(0) // 2, :]
        # logf2 = logf[logf.size(0) // 2:, :]
        # print(logf)
        mlogf = logf.mean(dim=0)
        # sumf1 = mlogf1.sum()
        # mlogf2 = logf2.mean(dim=0)
        # sumf2 = mlogf2.sum()
        # l = mlogf.sum()
        return -mlogf.sum()

