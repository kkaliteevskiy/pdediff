import torch
from torch import tensor
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal
from tqdm import tqdm
from pdediff.sde import VPSDE


# define globals
global A, T, alpha, diff_steps, ts, sigmas2_total, del_sigmas2, del_sigmas, sigma_y, dims, sde

diff_steps = 50
ts = torch.linspace(0, 1, diff_steps)
sigma_y = tensor(0.1)
dims = 1

# MMG params
sigma_0 = 0.5
mu1, mu2 = -1.0, 1.0

# define SDE
sde = VPSDE(tensor([1e-12]), tensor([1.0]).shape)

# observation operator
def obs(x):
    a = 1
    return a * torch.tanh(x/a)
A = obs

def sample_p0(n):
    x = torch.zeros(n)
    for i in range(n):
        if torch.rand(1) > 0.5:
            x[i] = torch.randn(1) * sigma_0 + mu1
        else:
            x[i] = torch.randn(1) * sigma_0 + mu2
    return x

def pt(x, t):
    mvn1 = MultivariateNormal(tensor([mu1 * sde.mu(t)]), (sde.mu(t)**2 * sigma_0**2 + sde.sigma(t)**2).reshape((1,1)))
    mvn2 = MultivariateNormal(tensor([mu2 * sde.mu(t)]), (sde.mu(t)**2 * sigma_0**2 + sde.sigma(t)**2).reshape((1,1)))
    return 0.5 * torch.exp(mvn1.log_prob(x)) + 0.5 * torch.exp(mvn2.log_prob(x))


def log_pt(x, t): # is this right?
    mvn1 = MultivariateNormal(tensor([mu1 * sde.mu(t)]), (sde.mu(t)**2 * sigma_0**2 + sde.sigma(t)**2).reshape((1,1)))
    mvn2 = MultivariateNormal(tensor([mu2 * sde.mu(t)]), (sde.mu(t)**2 * sigma_0**2 + sde.sigma(t)**2).reshape((1,1)))
    return torch.logsumexp(torch.stack([mvn1.log_prob(x) - torch.log(tensor(2.0)), mvn2.log_prob(x)- torch.log(tensor(2.0))], dim=-1), dim=-1)

def score_pt(xt, t): # xt[samples, dims]
    log_p = log_pt(xt, t)
    # log_p = torch.log(pt(xt, t))
    grads = torch.autograd.grad(log_p, xt, torch.ones_like(log_p))[0]
    return grads

def log_p_y_given_x(y, x):
    mvn = MultivariateNormal(A(x), (sigma_y**2).reshape(1,1))
    return mvn.log_prob(y)

def get_mu_from_xt(xt: tensor, t = tensor(0.0)):
    sc = score_pt(xt, t)
    if sc.isnan().any():
        print('nan in score')
    return  (xt + (1-sde.alpha(t))* sc)/sde.mu(t)

def get_guidance(xt, y, t, n_samples = 100):
    '''Vanilla MC guidance estimation'''
    mu = get_mu_from_xt(xt, t)
    if mu.isnan().any():
        print('nan in mu')
    # for VPSDE Var[x0|xt] = (1-alpha_bar)/sqrt(alpha_bar)
    sigma2 = (1-sde.alpha(t))/sde.mu(t)
    sigma = sigma2**0.5
    x_hats = mu + sigma * torch.randn(n_samples, *mu.shape)
    if x_hats.isnan().any():
        print('nan in x_hats')
    log_p_y_ests = log_p_y_given_x(y, x_hats)
    log_p_est = torch.logsumexp(log_p_y_ests, dim=0)
    grads = torch.autograd.grad(log_p_est, xt, torch.ones_like(log_p_est))[0]
    if grads.isnan().any():
        print('nan in grads')
    return grads, x_hats

def get_cov_from_xt(xt, t): #NOTE: this isn't used any more?
    log_p = torch.log(pt(xt, t))
    grads_1 = torch.autograd.grad(log_p, xt, torch.ones_like(log_p), create_graph=True)[0]
    grads_2 = torch.autograd.grad(grads_1, xt, torch.ones_like(grads_1))[0]
    return grads_2 #  sigmas2_total[t] + sigmas2_total[t]**2 * 


def log_p_x0_given_y_xt(y, x0, xt, t):
    ''' compute the probability of of the proposal function q(x0|xt, y)'''
    mu_tweedie = get_mu_from_xt(xt, t).detach().requires_grad_(True)
    Amu = A(mu_tweedie)
    partial_A = torch.autograd.grad(Amu, mu_tweedie, torch.ones_like(Amu))[0] # partial derivative of A wrt mu
    mu = mu_tweedie + (y - Amu) / partial_A
    var = (sigma_y/partial_A)**2
    log_prob = torch.zeros(x0.shape[:-1])
    for i in range(mu.shape[0]):
        mvn = MultivariateNormal(mu[i], var[i].reshape((1,1)))
        log_prob[:, i] = mvn.log_prob(x0[:, i])
    return log_prob
 

def sample_x0_given_y_xt(y, xt, t, n_samples = 10):
    mu_tweedie = get_mu_from_xt(xt, t).detach().requires_grad_(True) #tensor([1.0]).detach().requires_grad_(True) # 
    Amu = A(mu_tweedie)
    partial_A = torch.autograd.grad(Amu, mu_tweedie, torch.ones_like(Amu))[0] # partial derivative of A wrt mu

    # y = Amu + partial_A * (xt - mu) + sigma_y * eta
    # x = mu + y / part_A - Amu / partial_A   +   sigma_y / partial_A * eta 

    mu = mu_tweedie + (y - Amu) / partial_A
    std = sigma_y / partial_A
    x_hats = mu + std * torch.randn(n_samples, *mu.shape)
    return x_hats

def log_p_tweedie(x0, xt, t):
    '''compute the log probability of x0 given xt
    Posterior p(x0|xt) is approximated as Gaussian with variance sigma^2(t)'''

    mu = get_mu_from_xt(xt, t)
    var = ((1-sde.alpha(t))/sde.mu(t)).reshape(1,1)
    mvn = MultivariateNormal(mu, var)
    return mvn.log_prob(x0)

def get_guidance_IS(y, xt, t, n_samples = 1):
    '''
    Importance sampling guidance estimation
    Linearisation is done about the Tweedie mean
    '''

    x_hats = sample_x0_given_y_xt(y, xt, t, n_samples)
    log_p_x0_given_xt_sampled = log_p_tweedie(x_hats, xt, t) # seems legit
    log_p_x0_given_y_xt_sampled = log_p_x0_given_y_xt(y, x_hats, xt, t) # needs to be rewritten
    log_p_y_given_x_hats_sampled = log_p_y_given_x(y, x_hats) # p(y|x0_hat) to be estimated
    log_importance_estimates = log_p_y_given_x_hats_sampled - log_p_x0_given_y_xt_sampled + log_p_x0_given_xt_sampled
    log_prob = torch.logsumexp(log_importance_estimates, dim=0)
    grads = torch.autograd.grad(log_prob, xt, torch.ones_like(log_prob))[0]

    return grads, x_hats



