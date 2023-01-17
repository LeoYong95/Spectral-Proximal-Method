import math
from typing import List, Optional

import torch
from torch.optim.optimizer import Optimizer

from .types import Betas2, OptFloat, OptLossClosure, Params
from decimal import Decimal

Grads = Params

__all__ = ('SP')


class SP(Optimizer):
    r"""Implements Spectral Proximal Method Algorithm.
    It has been proposed in `Optimized Machine Learning Algorithm using Hybrid Proximal Method with Spectral Gradient Techniques`.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.01)
        
        sigma (float, optional): sparsity variable (default: 0.1)

        momentum (float, optional): momentum variable (default: 0.9)

        sparsity (bool, optional): sparsity status variable (default: True)

        nesterov (bool, optional): nesterov status variable (default: False)

        Example:
        >>> import optimizer as optim
        >>> optimizer = optim.SP(model.parameters(), lr = 1.0, sigma = 0.1, momentum = 0.9, sparsity = True, nesterov = true)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward(create_graph=True)
        >>> optimizer.step()

        Note:
            Reference code: https://github.com/jettify/pytorch-optimizer.git
            Paper: https://doi.org/10.22541/au.167350875.57067000/v1
    """

    def __init__(
        self,
        params: Params,
        lr: float = 0.01,
        sigma: float = 0.1,
        momentum: float = 0.9 ,
        sparsity: bool = True,
        nesterov: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= sigma <= 1.0:
            raise ValueError(
                'Invalid sigma value: {}'.format(hessian_power)
            )
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(
            lr=lr,
            sigma=sigma,
            momentum=momentum,
            sparsity=sparsity,
            nesterov=nesterov,

        )
        super(SP, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """Perform a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                #get parameters
                grads = p.grad.data
                para = p.data.clone()
                gra = grads.clone()
                if p.grad.is_sparse:
                    raise RuntimeError('SP does not support sparse gradients')

                #state initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['params_prev'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['grads_prev'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['sgd'] = True
                    state['buffer_prev'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                else:
                    state['sgd'] = False
                
                params_prev, grads_prev, sgd, buffer_prev = (state['params_prev'],state['grads_prev'],state['sgd'], state['buffer_prev'])

                state['step'] += 1

                if sgd:
                    #first iteration
                    buffer_prev = grads.clone()
                    p.data.add_(grads,alpha=-group['lr'])
                else:

                    sk = para.sub_(params_prev)
                    yk = gra.sub_(grads_prev)

                    #update previous params and gradients
                    params_prev = p.data.clone()
                    grads_prev = grads.clone()
                    
                    yk_sk_sum = torch.sum(torch.flatten(yk.mul_(sk)))
                    sk_sqr_sum = torch.sum(torch.flatten(torch.square(sk)))
                    sk_fourth_sum = torch.sum(torch.flatten(torch.pow(sk,4)))

                    #get damping matrix
                    if sk_sqr_sum > yk_sk_sum:
                        b_inv = torch.square(sk).mul_(sk_sqr_sum.sub_(yk_sk_sum).div_(sk_fourth_sum)).add_(1)
                        h_mat = 1/b_inv
                    else:

                        #check if yk_sk_sum is nan
                        yk_sk_sum = torch.nan_to_num(yk_sk_sum, nan=1.0, posinf=1.0, neginf=-1.0)
                        b_inv = torch.square(sk).div_(yk_sk_sum)
                        h_mat = 1/b_inv
                    
                    if group['nesterov']:

                        #add Nesterov momentum modifier
                        grads = grads.add_(buffer_prev.mul_(group['momentum']).add_(grads), alpha=group['momentum'])

                    p.data.addcmul_(b_inv,grads,value=-group['lr'])

                    if group['sparsity']:

                        #get saliency matrix
                        sal = h_mat.mul_((torch.square(sk).div_(2)).mul_(torch.square(para).div_(2)))
                        thresh=torch.quantile(torch.flatten(sal),group['sigma'],dim=0,keepdim=True,interpolation='midpoint')
               
                        #sparse operator 
                        for (pr, s) in zip(p.data, sal):
                            pr = torch.where(s>thresh,pr.type(torch.DoubleTensor).to('cuda'),0.)
                
        return loss


        
                
