#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#normal MAML, also contain the adv accuracy calculation
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
from torch.autograd import Variable
import  numpy as np

from    learner import Learner
from    copy import deepcopy
from attack import PGD
from Visualization import RobustVis




class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config, device):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.device = device


        self.net = Learner(config, args.imgc, args.imgsz)
        #self.netadv = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        #self.meta_optimadv = optim.Adam(self.netadv.parameters(), lr=self.meta_lr)




    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry, x_nat):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        
        need_adv = False
        #AT
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.update_lr, momentum=0.9, weight_decay=5e-4)
        eps, step = (2.0,10)
        losses_q_adv = [0 for _ in range(self.update_step + 1)]
        corrects_adv = [0 for _ in range(self.update_step + 1)]
        
        


        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            
            #PGD AT
            if need_adv:
                at = PGD(eps=eps / 255.0, sigma=2 / 255.0, nb_iter=step)
                data = x_qry[i]
                label = y_qry[i]
                optimizer.zero_grad()
                adv_inp_adv = at.attack(self.net, fast_weights, data, label)
                optimizer.zero_grad()
                self.net.train()

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct
                
                
            #PGD AT
            if need_adv:
                data = x_qry[i]
                label = y_qry[i]
                optimizer.zero_grad()
                adv_inp = at.attack(self.net, self.net.parameters(), data, label)
                optimizer.zero_grad()
                self.net.train()
                with torch.no_grad():
                    logits_q_adv = self.net(adv_inp, self.net.parameters(), bn_training=True)
                    loss_q_adv = F.cross_entropy(logits_q_adv, label)
                    losses_q_adv[0] += loss_q_adv

                    pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                    correct_adv = torch.eq(pred_q_adv, label).sum().item()
                    corrects_adv[0] = corrects_adv[0] + correct_adv

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct
                
                
                
                #PGD AT
                if need_adv:
                    logits_q_adv = self.net(adv_inp_adv, fast_weights, bn_training=True)
                    loss_q_adv = F.cross_entropy(logits_q_adv, label)
                    losses_q_adv[1] += loss_q_adv

                    pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                    correct_adv = torch.eq(pred_q_adv, label).sum().item()
                    corrects_adv[1] = corrects_adv[1] + correct_adv
                

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q
                
                
                #PGD AT
                if need_adv:
                    at = PGD(eps=eps / 255.0, sigma=2 / 255.0, nb_iter=step)
                    data = x_qry[i]
                    label = y_qry[i]
                    optimizer.zero_grad()
                    adv_inp_adv = at.attack(self.net, fast_weights, data, label)
                    optimizer.zero_grad()
                    
                    logits_q_adv = self.net(adv_inp_adv, fast_weights, bn_training=True)
                    loss_q_adv = F.cross_entropy(logits_q_adv, label)
                    losses_q_adv[k + 1] += loss_q_adv
                

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct
                    
                    #PGD AT
                    if need_adv:
                        pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                        correct_adv = torch.eq(pred_q_adv, label).sum().item()
                        corrects_adv[k + 1] = corrects_adv[k + 1] + correct_adv



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        
        loss_q_adv = losses_q_adv[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()

        self.meta_optim.step()
        


        accs = np.array(corrects) / (querysz * task_num)
        accs_adv = np.array(corrects_adv) / (querysz * task_num)

        return accs, accs_adv


    def finetunning(self, x_spt, y_spt, x_qry, y_qry, x_ss, num_ind, kwargs):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        
        need_adv = True
        beta = 0
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.update_lr, momentum=0.9, weight_decay=5e-4)
        eps, step = (2,10)
        corrects_adv = [0 for _ in range(self.update_step_test + 1)]
        corrects_adv_prior = [0 for _ in range(self.update_step_test + 1)]
        

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
        
        
        
        #PGD AT
        if need_adv:
            at = PGD(eps=eps / 255.0, sigma=2 / 255.0, nb_iter=step)
            data = x_qry
            label = y_qry
            optimizer.zero_grad()
            adv_inp_adv = at.attack(net, fast_weights, data, label)
        
        
        

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            
            #find the correct index
            corr_ind = (torch.eq(pred_q, y_qry) == True).nonzero()
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct
            
            
        #PGD AT
        if need_adv:
            data = x_qry
            label = y_qry
            optimizer.zero_grad()
            adv_inp = at.attack(net, net.parameters(), data, label)
            with torch.no_grad():
                logits_q_adv = net(adv_inp, net.parameters(), bn_training=True)
                pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                correct_adv = torch.eq(pred_q_adv, label).sum().item()
                correct_adv_prior = torch.eq(pred_q_adv[corr_ind], label[corr_ind]).sum().item()
                corrects_adv[0] = corrects_adv[0] + correct_adv
                corrects_adv_prior[0] = corrects_adv_prior[0] + correct_adv_prior/len(corr_ind)

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            #find the correct index
            corr_ind = (torch.eq(pred_q, y_qry) == True).nonzero()
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct
            
            
            #PGD AT
            if need_adv:
                logits_q_adv = net(adv_inp_adv, fast_weights, bn_training=True)
                pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                correct_adv = torch.eq(pred_q_adv, label).sum().item()
                correct_adv_prior = torch.eq(pred_q_adv[corr_ind], label[corr_ind]).sum().item()
                corrects_adv[1] = corrects_adv[1] + correct_adv
                corrects_adv_prior[1] = corrects_adv_prior[1] + correct_adv_prior/len(corr_ind)
            

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)
            
            
            
            #PGD AT
            if need_adv:
                at = PGD(eps=eps / 255.0, sigma=2 / 255.0, nb_iter=step)
                data = x_qry
                label = y_qry
                optimizer.zero_grad()
                adv_inp_adv = at.attack(net, fast_weights, data, label)

                logits_q_adv = net(adv_inp_adv, fast_weights, bn_training=True)
                loss_q_adv = F.cross_entropy(logits_q_adv, label)
            
            

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                #find the correct index
                corr_ind = (torch.eq(pred_q, y_qry) == True).nonzero()
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct
                
                
                #PGD AT
                if need_adv:
                    pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                    correct_adv = torch.eq(pred_q_adv, label).sum().item()
                    correct_adv_prior = torch.eq(pred_q_adv[corr_ind], label[corr_ind]).sum().item()
                    corrects_adv[k + 1] = corrects_adv[k + 1] + correct_adv
                    corrects_adv_prior[k + 1] = corrects_adv_prior[k + 1] + correct_adv_prior/len(corr_ind)


        robustmod = RobustVis(net, self.device)
        loss_p, x_p = robustmod.forward(x_ss.to(self.device), num_ind, **kwargs)
        
        
        del net

        accs = np.array(corrects) / querysz
        
        accs_adv = np.array(corrects_adv) / querysz
        
        accs_adv_prior = np.array(corrects_adv_prior)

        return loss_p, x_p



def main():
    pass


if __name__ == '__main__':
    main()


