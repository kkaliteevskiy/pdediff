import torch
from torch import tensor
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import MultivariateNormal, Normal
from scipy.stats import multivariate_normal
from tqdm import tqdm
from pdediff.sde import VPSDE


# define globals
global A, T, alpha, diff_steps, ts, sigmas2_total, del_sigmas2, del_sigmas, sigma_y, dims, sde

diff_steps = 100
ts = torch.linspace(0, 1, diff_steps)
sigma_y = tensor(0.1)
dims = 1

# MMG params
sigma_0 = 0.5
mu1, mu2 = -1, +1 #  shifted by 10

# define SDE
sde = VPSDE(tensor([1e-12]), tensor([1.0]).shape)

# observation operator
def obs(x):
    a = 1
    return a * torch.tanh((x)/a)
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
    mvn1 = Normal(tensor([mu1 * sde.mu(t)]), (sde.mu(t)**2 * sigma_0**2 + sde.sigma(t)**2).sqrt())
    mvn2 = Normal(tensor([mu2 * sde.mu(t)]), (sde.mu(t)**2 * sigma_0**2 + sde.sigma(t)**2).sqrt())
    return 0.5 * torch.exp(mvn1.log_prob(x)) + 0.5 * torch.exp(mvn2.log_prob(x))


def log_pt(x, t): # is this right?
    mvn1 = Normal(tensor([mu1 * sde.mu(t)]), (sde.mu(t)**2 * sigma_0**2 + sde.sigma(t)**2).sqrt())
    mvn2 = Normal(tensor([mu2 * sde.mu(t)]), (sde.mu(t)**2 * sigma_0**2 + sde.sigma(t)**2).sqrt())
    return torch.logsumexp(torch.stack([mvn1.log_prob(x) - torch.log(tensor(2.0)), mvn2.log_prob(x)- torch.log(tensor(2.0))], dim=-1), dim=-1)

def score_pt(xt, t): # xt[samples, dims]
    log_p = log_pt(xt, t)
    # log_p = torch.log(pt(xt, t))
    grads = torch.autograd.grad(log_p, xt, torch.ones_like(log_p))[0]
    return grads

def log_p_y_given_x(y, x, sigma_y):
    mvn = Normal(A(x), sigma_y)
    return mvn.log_prob(y)
    # mvn = MultivariateNormal(A(x), (sigma_y**2).reshape(1,1))
    # return mvn.log_prob(y)

def get_mu_from_xt(xt: tensor, t = tensor(0.0)):
    sc = score_pt(xt, t)
    # if sc.isnan().any():
    #     print('nan in score')
    return  (xt + (1-sde.alpha(t))* sc)/sde.mu(t) # same as (xt + sde.sigma(t)**2 * sc)/sde.mu(t)

def get_guidance(xt, y, t, n_samples = 100, sigma_y = sigma_y):
    '''Vanilla MC guidance estimation'''
    mu = get_mu_from_xt(xt, t)
    # if mu.isnan().any():
    #     print('nan in mu')
    # for VPSDE Var[x0|xt] = (1-alpha_bar)/sqrt(alpha_bar)

    # sigma2 = (1-sde.alpha(t))/sde.mu(t) # why is sigma2 computed this way and not sde.sigma(t)**2??
    # also pretty sure Var[x0|xt] = sigma**2/mu**2
    # pretty sure this is correct but version above was used before: sigma2 = sde.sigma(t)**2 / sde.mu(t)**2
    sigma =  sde.sigma(t)/sde.mu(t)# sigma2**0.5
    x_hats = mu + sigma * torch.randn(n_samples, *mu.shape)
    # if x_hats.isnan().any():
    #     print('nan in x_hats')
    log_p_y_ests = log_p_y_given_x(y, x_hats, sigma_y)
    log_p_est = torch.logsumexp(log_p_y_ests, dim=0)
    grads = torch.autograd.grad(log_p_est, xt, torch.ones_like(log_p_est))[0]
    # if grads.isnan().any():
    #     print('nan in grads')
    return grads, x_hats

def get_cov_from_xt(xt, t): #NOTE: this isn't used any more?
    log_p = torch.log(pt(xt, t))
    grads_1 = torch.autograd.grad(log_p, xt, torch.ones_like(log_p), create_graph=True)[0]
    grads_2 = torch.autograd.grad(grads_1, xt, torch.ones_like(grads_1))[0]
    return grads_2 #  sigmas2_total[t] + sigmas2_total[t]**2 * 


def log_p_x0_given_y_xt(y, x0, xt, t, sigma_y = sigma_y): # shape x0 = [MC_sample, n_smaples, dims]
    ''' compute the probability of of the proposal function q(x0|xt, y)'''
    mu_tweedie = get_mu_from_xt(xt, t).detach().requires_grad_(True)
    Amu = A(mu_tweedie)
    partial_A = torch.autograd.grad(Amu, mu_tweedie, torch.ones_like(Amu))[0] # partial derivative of A wrt mu
    mu = mu_tweedie + (y - Amu) / partial_A
    std = (sigma_y/partial_A)
    var = std**2
    # log_prob = torch.zeros(x0.shape[:-1]) # truncate dimention index [n_smaple, mc_sample]
    # for i in range(mu.shape[0]): # for each sample
    #     mvn = MultivariateNormal(mu[i], var[i].reshape((1,1)))
    #     log_prob[:, i] = mvn.log_prob(x0[:, i])
    mvn = Normal(mu, std)
    log_prob = mvn.log_prob(x0).squeeze(-1)

    return log_prob
 

def sample_x0_given_y_xt(y, xt, t, n_samples = 10, sigma_y = sigma_y):  
    
    mu_tweedie = get_mu_from_xt(xt, t).detach().requires_grad_(True) #tensor([1.0]).detach().requires_grad_(True) # 
    Amu = A(mu_tweedie)
    partial_A = torch.autograd.grad(Amu, mu_tweedie, torch.ones_like(Amu))[0] # partial derivative of A wrt mu

    # y = Amu + partial_A * (xt - mu) + sigma_y * eta
    # x = mu + y / part_A - Amu / partial_A   +   sigma_y / partial_A * eta 

    mu = mu_tweedie + (y - Amu) / partial_A
    std = sigma_y / partial_A
    x_hats = mu + std * torch.randn(n_samples, *mu.shape)
    mvn = Normal(mu, std)
    log_p = mvn.log_prob(x_hats).sum(-1) # sum over dims
    return x_hats, log_p

def log_p_tweedie(x0, xt, t):
    '''compute the log probability of x0 given xt
    Posterior p(x0|xt) is approximated as Gaussian with variance sigma^2(t)'''

    mu = get_mu_from_xt(xt, t)
    std = sde.sigma(t)/sde.mu(t) # sigma2**0.5
    # var = ((1-sde.alpha(t))/sde.mu(t))#.reshape(1,1)
    # mvn = MultivariateNormal(mu, var)

    mvn = Normal(mu, std)
    return mvn.log_prob(x0).sum(-1)

def get_guidance_IS(y, xt, t, n_samples = 1, sigma_y = sigma_y):
    '''
    Importance sampling guidance estimation
    Linearisation is done about the Tweedie mean
    '''

    x_hats, log_p_proposal = sample_x0_given_y_xt(y, xt, t, n_samples, sigma_y)
    log_p_x0_given_xt_sampled = log_p_tweedie(x_hats, xt, t) # seems legit
    # log_p_proposal = log_p_x0_given_y_xt(y, x_hats, xt, t, sigma_y) # needs to be rewritten
    log_p_likelihood = log_p_y_given_x(y, x_hats, sigma_y).sum(-1) # p(y|x0_hat) NOTE: sum over dims
    log_importance_estimates = log_p_likelihood - log_p_proposal + log_p_x0_given_xt_sampled
    log_prob = torch.logsumexp(log_importance_estimates, dim=0)
    grads = torch.autograd.grad(log_prob, xt, torch.ones_like(log_prob))[0]

    return grads, x_hats


def log_p_x0_given_y_xt_2(y, x0, xt, t, sigma_y):
    mu_tweedie = get_mu_from_xt(xt, t).detach().requires_grad_(True) #tensor([1.0]).detach().requires_grad_(True) # 
    Amu = A(mu_tweedie)
    partial_A = torch.autograd.grad(Amu, mu_tweedie, torch.ones_like(Amu))[0] # partial derivative of A wrt mu

    # Gaussian 1 from linearilisation
    mu1 = mu_tweedie + (y - Amu) / partial_A # Gaussian 1 mean
    std1 = sigma_y / partial_A # Gaussian 1 std

    # Gaussian 2 from Tweedie
    mu2 = get_mu_from_xt(xt, t) # Gaussian 2 mean
    std2 = sde.sigma(t) # Gaussian 2 std

    #product of two Gaussians
    var = (std1**-2 + std2**-2)**-1
    mu = var * (mu1 / std1**2 + mu2 / std2**2)
    std = var**0.5
    mvn = Normal(mu, std)

    return mvn.log_prob(x0).sum(-1) # sum over dims
    


def sample_x0_given_y_xt_2(y, xt, t, n_samples = 10, sigma_y = sigma_y):  
    '''Imortance sampling 2  proposal
    Returns x_hats and log_p from proposal'''
    mu_tweedie = get_mu_from_xt(xt, t).detach().requires_grad_(True) #tensor([1.0]).detach().requires_grad_(True) # 
    Amu = A(mu_tweedie)
    partial_A = torch.autograd.grad(Amu, mu_tweedie, torch.ones_like(Amu))[0] # partial derivative of A wrt mu

    # Gaussian 1 from linearilisation
    mu1 = mu_tweedie + (y - Amu) / partial_A # Gaussian 1 mean
    std1 = sigma_y / partial_A # Gaussian 1 std

    # Gaussian 2 from Tweedie
    mu2 = get_mu_from_xt(xt, t) # Gaussian 2 mean
    std2 = sde.sigma(t) # Gaussian 2 std

    #product of two Gaussians
    var = (std1**-2 + std2**-2)**-1
    mu = var * (mu1 / std1**2 + mu2 / std2**2)
    std = var**0.5
    mvn = Normal(mu, std)

    x_hats = mu + std * torch.randn(n_samples, *mu.shape)
    log_p = mvn.log_prob(x_hats)
    
    return x_hats, log_p


def get_guidance_IS_2(y, xt, t, n_samples = 1, sigma_y = sigma_y):
    '''
    Importance sampling guidance estimation
    Linearisation is done about the Tweedie mean
    proposal is the linearied Gaussian 
    multiplied by the tweedie posterior gaussian
    '''
    x_hats, log_p_proposal = sample_x0_given_y_xt_2(y, xt, t, n_samples, sigma_y) # .sum(-1)
    log_p_proposal = log_p_proposal.sum(-1) # sum over dims
    log_p_target = log_p_tweedie(x_hats, xt, t) # seems legit
    log_p_likelihood = log_p_y_given_x(y, x_hats, sigma_y).sum(-1) # .sum(-1)
    log_prob = log_p_likelihood - log_p_proposal + log_p_target
    log_prob = torch.logsumexp(log_prob, dim=0)
    grads = torch.autograd.grad(log_prob, xt, torch.ones_like(log_prob))[0]
    return grads , x_hats
    


### Target Distributions


def get_targert_hist(y, sigma_y, n_bins):
    x_ = torch.linspace(-10, 10, 10000)
    dx = x_[1] - x_[0]
    log_p_x = log_pt(x_, ts[0]).squeeze(-1)
    log_py_x = log_p_y_given_x(y * torch.ones_like(x_), x_, sigma_y)
    log_px_y = log_py_x + log_p_x
    px_y = torch.exp(log_px_y)
    px_y = (px_y[:-1] + px_y[1:])/2 * dx
    pdf = px_y/px_y.sum()/dx

    # compute the 99.8 % interval
    cdf = pdf.cumsum(axis = 0) * dx
    xmin = x_[torch.where(cdf >= 0.00001)[0][0]]
    xmax = x_[torch.where(cdf >= 0.99999)[0][0]]
    # xmin, xmax = -4, 4
    # print(xmin, xmax)



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

def posterior_pdf(x, y, sigma_y):
    x_ = torch.linspace(-10, 10, 10000) # compute normalization constant
    dx = x_[1] - x_[0]
    log_p_x = log_pt(x_, ts[0]).squeeze(-1)
    log_py_x = log_p_y_given_x(y * torch.ones_like(x_), x_, sigma_y)
    log_px_y = log_py_x + log_p_x
    px_y = torch.exp(log_px_y)

    Z = px_y.sum()*dx
    return torch.exp(log_p_y_given_x(y, x, sigma_y)) * pt(x, ts[0]) / Z





