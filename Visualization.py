#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
# import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import dill
import os
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

# from . import helpers
# from . import attack_steps
from learner import Learner

class RobustVis(torch.nn.Module):

    def __init__(self, model, device):

        super(RobustVis, self).__init__()
        #self.normalize = helpers.InputNormalize(dataset.mean, dataset.std)
        configtest = [
            ('conv2d', [32, 3, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 1, 0]),
            ('flatten', [])
        ]
        
        copymod = Learner(configtest, 3, 84)#.to('cuda:3')
        for i in range(0,16):
            copymod.parameters()[i] = model.parameters()[i]
        
        
        self.model = copymod.to(device)
        self.model.eval()
        self.device = device

    def forward(self, x, target, *_, constraint, eps, step_size, iterations, criterion,
                random_start=False, random_restarts=False, do_tqdm=False,
                targeted=False, custom_loss=None, should_normalize=False, 
                orig_input=None, use_best=False, sigma=0.000001):

        
        # Can provide a different input to make the feasible set around
        # instead of the initial point

        if orig_input is None: orig_input = x.detach()
        orig_input = orig_input.to(self.device)

        # Multiplier for gradient ascent [untargeted] or descent [targeted]
        m = -1 if targeted else 1

        # Initialize step class
#         step = STEPS[constraint](eps=eps, orig_input=orig_input, step_size=step_size)

        def calc_loss(inp, index):

#             if should_normalize:
#                 inp = self.normalize(inp)
            output_vec = self.model(inp)
            output = [output_vec[i][index] for i in range(output_vec.size()[0])]
#             if custom_loss:
#                 return custom_loss(self.model, inp, target)

            return output


        def get_pert_examples(x):

            pert = torch.empty(x.shape).normal_(mean=0,std=sigma).to(self.device)
            #print(torch.max(pert).item())
            # Random start (to escape certain types of gradient masking)
            if random_start:
                x = torch.clamp(x + step.random_perturb(x), 0, 1)

            iterator = range(iterations)
            if do_tqdm: iterator = tqdm(iterator)

            # Keep track of the "best" (worst-case) loss and its
            # corresponding input
            best_loss = None
            best_x = None

            # A function that updates the best loss and best input
            def replace_best(loss, bloss, x, bx):
                if bloss is None:
                    bx = x.clone().detach()
                    bloss = losses.clone().detach()
                else:
                    replace = m * bloss < m * loss
                    bx[replace] = x[replace].clone().detach()
                    bloss[replace] = loss[replace]

                return bloss, bx

            delta = torch.zeros_like(x, requires_grad=True).requires_grad_(True)
            
            

#             W = torch.zeros(2048, requires_grad=False)
#             W[1858] = 1
            

            #
            x0 = x

#             step_d = STEPS[constraint](eps=eps, orig_input=delta, step_size=step_size)
#             step_m = attack_steps.LinfStep1(eps=eps, orig_input=M, step_size=step_size)

            #
            for _ in iterator:
                delta = delta.clone().detach().requires_grad_(True).to(self.device)

                x = x0 + pert + delta
                
                x = torch.clamp(x, 0, 1)
                losses = calc_loss(x, target)
                
                
#                 W1 = W.unsqueeze(0).expand(10, -1)

                #W1 = W
#                 losses = losses * W1

                loss = losses#torch.mean(losses)

                grad_d = torch.autograd.grad(loss, delta)
                #
#                 print(type(grad_d))
#                 print(len(grad_d))
#                 print(len(grad_d[0]))
#                 print(len(grad_d[0][0]))


                with torch.no_grad():
                    args = [losses, best_loss, x, best_x]
                    best_loss, best_x = replace_best(*args) if use_best else (losses, x)

                    delta = grad_d[0] * step_size + delta
#                     delta = step_d.project(delta)
                    
#                     #additional inf_norm constraint (for clean label attack)
#                     max_d = x0+20.0/255#torch.min(20/M.cpu().detach().numpy() + x0.cpu().detach().numpy(), eps)
#                     min_d = x0-20.0/255#torch.max(-20/M.cpu().detach().numpy() + x0.cpu().detach().numpy(), 0)
#                     delta = torch.where(delta > min_d, min_d, delta)
#                     delta = torch.where(delta < max_d, max_d, delta)

#                     M = step_m.make_step(grad_m) * m + M
#                     M = step_m.project(M, gamma)

#                     #weight method
#                     W = step_w.make_step(grad_w) * m + W
#                     W = step_w.project(W)
#                     #

                    if do_tqdm: iterator.set_description("Current loss: {l}".format(l=loss))
            

            
            
            
            x = x0 + delta
#             loss_ave = loss.mean(0)

            if not use_best: return losses, torch.clamp(x,0,1).clone().detach()
  

            losses = calc_loss(x, target)
            args = [losses, best_loss, x, best_x]
            best_loss, best_x = replace_best(*args)
            return best_loss, best_x



        # Random restarts: repeat the attack and find the worst-case
        # example for each input in the batch
        if random_restarts:
            to_ret = None

            orig_cpy = x.clone().detach()
            for _ in range(random_restarts):
                pert_loss, pertimg = get_pert_examples(orig_cpy)

                if to_ret is None:
                    to_ret = pertimg.detach()

                output = calc_loss(pertimg, target)
#                 corr, = helpers.accuracy(output, target, topk=(1,), exact=True)
#                 corr = corr.byte()
#                 misclass = ~corr
#                 to_ret[misclass] = adv[misclass]

            pert_ret = to_ret
        else:
            pert_loss, pert_ret = get_pert_examples(x)

        return pert_loss, pert_ret

# class AttackerModel(torch.nn.Module):
#     def __init__(self, model, dataset):
#         super(AttackerModel, self).__init__()
#         self.normalizer = helpers.InputNormalize(dataset.mean, dataset.std)
#         self.model = model
#         self.attacker = Attacker(model, dataset)

#     def forward(self, inp, target=None, make_adv=False, with_latent=False,
#                     fake_relu=False, with_image=True, **attacker_kwargs):
#         if make_adv:
#             assert target is not None
#             prev_training = bool(self.training)
#             self.eval()
#             adv = self.attacker(inp, target, **attacker_kwargs)
#             if prev_training:
#                 self.train()

#             inp = adv

#         if with_image:
#             normalized_inp = self.normalizer(inp)
#             output = self.model(normalized_inp, with_latent=with_latent,
#                                                     fake_relu=fake_relu)
#         else:
#             output = None

#         return (output, inp)
