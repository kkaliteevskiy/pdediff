import torch
from torch import tensor
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal
from tqdm import tqdm
import pdb

# define globals
global A, T, alpha, diff_steps, ts, sigmas2_total, del_sigmas2, del_sigmas, sigma_y, dims

T = tensor(10)
alpha = tensor(0.001)
diff_steps = 50
ts = torch.arange(0, diff_steps, dtype = torch.int32)
sigmas2_total = torch.logspace(torch.log10(alpha), torch.log10(T), diff_steps+1)
del_sigmas2 = torch.diff(sigmas2_total)
del_sigmas = torch.sqrt(del_sigmas2)
sigma_y = tensor(0.1).reshape(1,1)
dims = 1

# MMG params
sigma_0 = 0.5
mu1, mu2 = -1.0, 1.0

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
    var_t = sigmas2_total[t]
    mvn1 = MultivariateNormal(tensor([mu1]), (sigma_0**2 + var_t).reshape((1,1)))
    mvn2 = MultivariateNormal(tensor([mu2]), (sigma_0**2 + var_t).reshape((1,1)))
    return 0.5 * torch.exp(mvn1.log_prob(x)) + 0.5 * torch.exp(mvn2.log_prob(x))


def log_pt(x, t): # is this right?
    var_t = sigmas2_total[t]
    mvn1 = MultivariateNormal(tensor([-1.0]), (sigma_0**2 + var_t).reshape((1,1)))
    mvn2 = MultivariateNormal(tensor([1.0]), (sigma_0**2 + var_t).reshape((1,1)))
    # factors of 0.5 are due to the fact that the two modes are equally likely in the Mixture of Gaussians
    return torch.logsumexp(torch.stack([mvn1.log_prob(x) - torch.log(tensor(2.0)), mvn2.log_prob(x)- torch.log(tensor(2.0))], dim=-1), dim=-1)

def sample_pt(t, n):
    var1_t, var2_t = sigma_0**2 + sigmas2_total[t], sigma_0**2 + sigmas2_total[t]
    var1_t, var2_t = var1_t.reshape((1,1)), var2_t.reshape((1,1))
    x = torch.zeros(n)
    for i in range(n):
        if torch.rand(1) > 0.5:
            x[i] = torch.randn(1) * (var1_t)**0.5 + mu1
        else:
            x[i] = torch.randn(1) * (var2_t)**0.5 + mu2
    return x

def score_pt(xt, t): # xt[samples, dims]
    log_p = log_pt(xt, t)
    # log_p = torch.log(pt(xt, t))
    grads = torch.autograd.grad(log_p, xt, torch.ones_like(log_p))[0]
    return grads

def log_p_y_given_x(y, x):
    mvn = MultivariateNormal(A(x), sigma_y**2)
    return mvn.log_prob(y)

def get_targert_hist(y, sigma_y, n_bins):
    ''' compute the target histogram of x given y by numerically comuting the posterior p(x|y)'''
    x_ = torch.linspace(-3, 7, 10000)
    dx = x_[1] - x_[0]
    log_p_x = log_pt(x_, ts[0]).squeeze(-1)
    log_py_x = log_p_y_given_x(y * torch.ones_like(x_), x_, sigma_y)
    log_px_y = log_py_x + log_p_x
    px_y = torch.exp(log_px_y)
    px_y = (px_y[:-1] + px_y[1:])/2 * dx
    pdf = px_y/px_y.sum()/dx

    # compute the 99.8 % interval
    cdf = pdf.cumsum(axis = 0) * dx
    xmin = x_[torch.where(cdf >= 0.0001)[0][0]]
    xmax = x_[torch.where(cdf >= 0.9999)[0][0]]
    x_range = xmax - xmin
    Dx = float(x_range/n_bins)
    edges = np.linspace(xmin, xmax, n_bins+1)

    posterior_bar = torch.zeros(n_bins)
    for i in range(n_bins):
        x_ = torch.linspace(edges[i], edges[i+1], 101)
        # print(edges[i], edges[i+1])
        dx = x_[1] - x_[0]
        log_p_x = log_pt(x_, ts[0]).squeeze(-1)
        log_py_x = log_p_y_given_x(y * torch.ones_like(x_), x_, sigma_y)
        log_px_y = log_py_x + log_p_x
        px_y = torch.exp(log_px_y)
        posterior_bar[i] = (px_y[:-1] + px_y[1:]).sum()/(2)
    posterior_bar = posterior_bar/posterior_bar.sum()/Dx
    

    return posterior_bar, edges, Dx

def get_target_hist_with_tw(y, t, sigma_y, n_bins):
    ''' compute the analytic posterior p(x|y) 
    using the tweedie approximation for p(x0|xt)
    p(x0|y)
    '''
    pass

def get_mu_from_xt(xt: tensor, t = 1):
    sc = score_pt(xt, t)
    if sc.isnan().any():
        print('nan in score')
    return xt + sigmas2_total[t].item() * sc

def get_guidance(xt, y, t, n_samples = 100):
    '''Vanilla MC guidance estimation'''
    mu = get_mu_from_xt(xt, t)
    if mu.isnan().any():
        print('nan in mu')
    sigma = ((sigmas2_total[t])**0.5).reshape((1,1)) #  get_cov_from_xt(xt, t)**0.5 # 
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
    if var.isnan().any() or mu.isnan().any():
        pdb.set_trace()
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
    var = sigmas2_total[t].reshape((1,1))
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



