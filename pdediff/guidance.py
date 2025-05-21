from abc import ABC, abstractmethod
from typing import Callable, Optional, List
import numpy as np

import torch
from torch import Tensor, nn
from torch.func import grad_and_value, grad, vmap
from torch.distributions import Normal
from numpy.typing import ArrayLike
from pdediff.sde import VPSDE
from pdediff.utils.data_preprocessing import append_zeros

import pdb


######################################################
##################  Likelihood  ######################
######################################################


class Likelihood(ABC, nn.Module):
    """p(y|x_0) = p(y|A(x))"""

    def __init__(self, y: Tensor, A: Callable[[Tensor], Tensor], mask: Tensor = None):
        super().__init__()
        self.register_buffer("y", y)
        self.A = A
        self.mask = mask

    def err(self, x: Tensor) -> Tensor:
        if self.mask is not None:
            obs = self.A(x, self.mask.to(x.device))
        else:
            obs = self.A(x)

        # assert obs.shape == self.y.shape, f"Obs shape {obs.shape} different to y shape {self.y.shape}"
        assert np.broadcast_shapes(obs.shape, self.y.shape) == obs.shape, f"Obs shape {obs.shape} and y shape {self.y.shape} are not broadcastable"
        return self.y.to(x.device) - obs

    def set_observation(self, y: Tensor, mask: Tensor = None):
        self.y.copy_(y)
        if mask is not None:
            self.mask = mask



class Gaussian(Likelihood):
    """p(y|x_0) = N(y|A(x), std^2)"""

    def __init__(self, y: Tensor, A: Callable[[Tensor], Tensor], std: float = 1.0, mask: Tensor = None):
        super().__init__(y, A, mask)
        self.std = std

    def sample(self, shape=()) -> Tensor:
        return Normal(self.y, self.std).rsample(shape)
    
    def log_prob(self, x: Tensor, sigma_t, mu_t, gamma = 0.1) -> Tensor:
        """Compute the log probability of the observation given the model output."""
        err = self.err(x)
        var = self.std ** 2 + gamma * (sigma_t / mu_t) ** 2
        return -0.5 * ((err ** 2) / var + torch.log(2 * torch.pi * var))


######################################################
################## Guidance term #####################
######################################################


class GuidedScore(ABC, nn.Module):
    def __init__(self, sde: VPSDE, likelihoods: Likelihood, N_MC_samples: Optional[int] = None):
        super().__init__()
        self.likelihoods = likelihoods
        self.sde = sde
        self.N_MC_samples = N_MC_samples

    def get_sde(self, shape) -> VPSDE:
        return self.sde.__class__(self, shape=shape)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class SDA(GuidedScore):
    def __init__(
        self,
        sde: VPSDE,
        likelihoods: List[Likelihood],
        gammas: List[float],
    ):
        super().__init__(sde, likelihoods)
        self.gammas = gammas
        assert len(likelihoods) == len(gammas), (
            f"Number of specified likelihoods {len(likelihoods)} is different "
            f"from the number of specified gammas {len(gammas)}. Please specify "
            "a guidance strength for each likelihood."
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor: # TODO: this is where teh computation happens 
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            # pdb.set_trace()
            x = x.detach().requires_grad_(True) # torch.Size([32, 9, 1, 256])

            eps = self.sde.noise_prediction_fn(x, t) # effectively the unconditional score, right??
            x_ = (x - sigma * eps) / mu

            log_probs = []
            for likelihood, gamma in zip(self.likelihoods, self.gammas):
                err = likelihood.err(x_)
                var = likelihood.std ** 2 + gamma * (sigma / mu) ** 2
                log_probs.append(-(err ** 2 / var).sum() / 2)

        for log_prob in log_probs:
            s, = torch.autograd.grad(log_prob, x, retain_graph=True)
            eps = eps - sigma * s # why is this premultiplied by sigma?
        return eps
    
### TODO : MC Based SDA  
class MC_SDA(GuidedScore):
    def __init__(
            self,
            sde: VPSDE,
            likelihoods: List[Likelihood],
            gammas: List[float],
            N_MC_samples: int = 1
        ):
            super().__init__(sde, likelihoods)
            self.gammas = gammas
            assert len(likelihoods) == len(gammas), (
                f"Number of specified likelihoods {len(likelihoods)} is different "
                f"from the number of specified gammas {len(gammas)}. Please specify "
                "a guidance strength for each likelihood."
            )
            self.N_MC_samples = N_MC_samples
        
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            eps = self.sde.noise_prediction_fn(x, t) # effectively the unconditional score, right??
            x_ = (x - sigma * eps) / mu # tweedie mean
            # device = x.device

            log_probs = []
            if len(self.likelihoods) == 1: # conditioning only on the initial consitions, no observations
                for likelihood, gamma in zip([self.likelihoods[0]], [self.gammas[0]]):
                    err = likelihood.err(x_)
                    var = likelihood.std ** 2 + gamma * (sigma / mu) ** 2
                    log_probs.append(-(err ** 2 / var).sum() / 2)

                # apply guidance
                for log_prob in log_probs:
                    s, = torch.autograd.grad(log_prob, x, retain_graph=True)
                    eps = eps - sigma * s # why is this premultiplied by sigma?


            elif len(self.likelihoods) == 2:
                # conditioning on the initial conditions and observations
                # first likelihood is the observation likelihood
                # second is the AR conditioning

                # pdb.set_trace()
                # AR conditioning likelihoods - effectively regular SDA
                for likelihood, gamma in zip([self.likelihoods[1]], [self.gammas[1]]):
                    err = likelihood.err(x_)
                    var = likelihood.std ** 2 + gamma * (sigma / mu) ** 2
                    log_probs.append(-(err ** 2 / var).sum() / 2)
                
                for log_prob in log_probs:
                    s, = torch.autograd.grad(log_prob, x, retain_graph=True)
                    eps = eps - sigma * s # why is this premultiplied by sigma?


                # apply guidance based on observation samples
                # generate samples
                log_probs_guidance = []
                for likelihood, gamma in zip([self.likelihoods[0]], [self.gammas[0]]):
                    x_hats = x_.repeat(self.N_MC_samples, *([1] * (x_.dim())))
                    x_hats = x_hats + torch.randn_like(x_hats) * sigma / mu # add stability constant
                    log_p_y_given_x_i = torch.zeros_like(x_hats)

                    # for each sample, compute p(y|x_i)
                    for i in range(self.N_MC_samples):
                        # pdb.set_trace()
                        log_p_y_given_x_i[i] = self.likelihoods[0].log_prob(x_hats[i], sigma, mu, gamma = 0.05) # tune gamma

                    # pdb.set_trace()
                    # normaliser = 1 /(1 + (sigma / mu) ** 2)
                    log_p_y_given_xt = torch.logsumexp(log_p_y_given_x_i, dim = 0).sum() # * normaliser #  normalisation factor similar to SDA
                    log_probs_guidance.append(log_p_y_given_xt)

                # apply guidance
                for log_prob in log_probs_guidance:
                    s, = torch.autograd.grad(log_prob, x, retain_graph=True)
                    eps = eps - sigma * s # why is this premultiplied by sigma?

        return eps


class MC_SDA_IS(GuidedScore):
    def __init__(
            self,
            sde: VPSDE,
            likelihoods: List[Likelihood],
            gammas: List[float],
            N_MC_samples: int = 1
        ):
            super().__init__(sde, likelihoods)
            self.gammas = gammas
            assert len(likelihoods) == len(gammas), (
                f"Number of specified likelihoods {len(likelihoods)} is different "
                f"from the number of specified gammas {len(gammas)}. Please specify "
                "a guidance strength for each likelihood."
            )
            self.N_MC_samples = N_MC_samples
        
    def sample_x_hats(self, y: Tensor, x_: Tensor, t: Tensor, n_smples=1, sigma_y=1.0) -> Tensor:
        # Returns x_hats: [n_smples, *x.shape]
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)
        x_hats = torch.zeros((n_smples, *x_.shape), device=x_.device, dtype=x_.dtype)
        mu_tweedie = x_ # tweedie mean  
        Amu = self.likelihoods[0].A(mu_tweedie, self.likelihoods[0].mask.to(x_.device)).requires_grad_(True)
        partial_A = torch.autograd.grad(Amu, mu_tweedie, torch.ones_like(mu_tweedie),retain_graph=True)[0] + 1e-6 # add small constant to avoid division by zero

        mu1 = mu_tweedie + (y - Amu) / partial_A 
        std1 = sigma_y / partial_A

        mu2 = mu_tweedie
        std2 = self.sde.sigma(t)

        var = (std1**-2 + std2**-2)**-1
        mu = var * (std1**-2 * mu1 + std2**-2 * mu2)
        std = var**0.5
        mvn = Normal(mu, std)

        x_hats = x_hats + torch.randn_like(x_hats) * std
        log_p = mvn.log_prob(x_hats) #  * self.likelihoods[0].mask.to(x_hats.device)

        return x_hats, log_p



        return None
        
    def log_p_tweedie(self, x: Tensor, t: Tensor, x_: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)
        x = x.detach().requires_grad_(True)
        mu_tweedie = x_
        std = sigma / mu
        mvn = Normal(mu_tweedie, std)
        log_p = mvn.log_prob(x)
        return log_p

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            eps = self.sde.noise_prediction_fn(x, t) # effectively the unconditional score, right??
            x_ = (x - sigma * eps) / mu # tweedie mean
            # device = x.device

            log_probs = []
            if len(self.likelihoods) == 1: # conditioning only on the initial consitions, no observations
                for likelihood, gamma in zip([self.likelihoods[0]], [self.gammas[0]]):
                    err = likelihood.err(x_)
                    var = likelihood.std ** 2 + gamma * (sigma / mu) ** 2
                    log_probs.append(-(err ** 2 / var).sum() / 2)

                # apply guidance
                for log_prob in log_probs:
                    s, = torch.autograd.grad(log_prob, x, retain_graph=True)
                    eps = eps - sigma * s # why is this premultiplied by sigma?


            elif len(self.likelihoods) == 2:
                # conditioning on the initial conditions and observations
                # first likelihood is the observation likelihood
                # second is the AR conditioning

                # pdb.set_trace()
                # AR conditioning likelihoods - effectively regular SDA
                for likelihood, gamma in zip([self.likelihoods[1]], [self.gammas[1]]):
                    err = likelihood.err(x_)
                    var = likelihood.std ** 2 + gamma * (sigma / mu) ** 2
                    log_probs.append(-(err ** 2 / var).sum() / 2)
                
                for log_prob in log_probs:
                    s, = torch.autograd.grad(log_prob, x, retain_graph=True)
                    eps = eps - sigma * s # why is this premultiplied by sigma?


                # apply guidance based on observation samples
                # generate samples
                log_probs_guidance = []
                for likelihood, gamma in zip([self.likelihoods[0]], [self.gammas[0]]):
                    y = likelihood.y.to(x.device)
                    sigma_y = likelihood.std
                    # TODO implement next 2 lines
                    # does everything need to be looped over for N_MC or can it be vectorised???
                    # pdb.set_trace()
                    x_hats, log_p_proposal = self.sample_x_hats(y, x_, t, n_smples = self.N_MC_samples, sigma_y = sigma_y)
                    x_hats = x_hats * likelihood.mask.to(x.device)
                    log_p_proposal = log_p_proposal * likelihood.mask.to(x.device)

                    log_p_target = torch.zeros_like(log_p_proposal)# function not vectorisable.. self.log_p_tweedie(x_hats, t)
                    
                    for i in range(self.N_MC_samples):
                        log_p_target[i] = self.log_p_tweedie(x_hats[i], t, x_)
                    
                    log_p_likelihood = likelihood.log_prob(x_hats, sigma, mu, gamma = 0.1)
                    log_prob = log_p_likelihood - log_p_proposal + log_p_target
                    log_prob = torch.logsumexp(log_prob, dim = 0)
                    log_prob = log_prob * likelihood.mask.to(x.device) # * self.likelihoods[0].mask.to(x.device)  
                    # log_prob[torch.isnan(log_prob)] = 0 
                    log_prob = log_prob.sum() # * normaliser #  normalisation factor similar to SDA
                    log_probs_guidance.append(log_prob)

                # apply guidance
                for log_prob in log_probs_guidance:
                    s, = torch.autograd.grad(log_prob, x, retain_graph=True)
                    eps = eps - sigma * s * self.likelihoods[0].mask.to(eps.device) # why is this premultiplied by sigma?

        return eps

class DPS(GuidedScore):
    def __init__(
        self,
        sde: VPSDE,
        likelihoods: List[Likelihood],
        gammas: List[float],
    ):
        super().__init__(sde, likelihoods)
        self.gammas = gammas
        assert len(likelihoods) == len(gammas), (
            f"Number of specified likelihoods {len(likelihoods)} is different "
            f"from the number of specified gammas {len(gammas)}. Please specify "
            "a guidance strength for each likelihood."
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # pdb.set_trace()
        mu, sigma = self.sde.mu(t), self.sde.sigma(t) # sde meand and std

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.noise_prediction_fn(x, t) # effectively the -score right??
            x_ = (x - sigma * eps) / mu # tweedie mean

            errs = []
            for likelihood in zip(self.likelihoods):
                errs.append((likelihood.err(x_)).square().sum())

        for err, gamma in zip(errs, self.gammas):
            s, = torch.autograd.grad(err, x, retain_graph=True)
            s = -s / gamma / err.sqrt()
            eps = eps - sigma * s

        return eps


class VideoDiff(GuidedScore):
    def __init__(
        self,
        sde: VPSDE,
        likelihoods: List[Likelihood],
        gammas: List[float],
    ):
        super().__init__(sde, likelihoods)
        self.gammas = gammas
        assert len(likelihoods) == len(gammas), (
            f"Number of specified likelihoods {len(likelihoods)} is different "
            f"from the number of specified gammas {len(gammas)}. Please specify "
            "a guidance strength for each likelihood."
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.noise_prediction_fn(x, t)
            x_ = (x - sigma * eps) / mu

            log_probs = []
            for likelihood, gamma in zip(self.likelihoods, self.gammas):
                err = likelihood.err(x_)
                var = gamma * (sigma / mu) ** 2
                log_probs.append(-(err ** 2 / var).sum() / 2)

        for log_prob in log_probs:
            s, = torch.autograd.grad(log_prob, x, retain_graph=True)
            eps = eps - sigma * s

        return eps

class PGDM(GuidedScore):
    def __init__(
        self,
        sde: VPSDE,
        likelihoods: List[Likelihood],
        gammas: List[float],
    ):
        super().__init__(sde, likelihoods)
        self.gammas = gammas
        assert len(likelihoods) == len(gammas), (
            f"Number of specified likelihoods {len(likelihoods)} is different "
            f"from the number of specified gammas {len(gammas)}. Please specify "
            "a guidance strength for each likelihood."
        )
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.noise_prediction_fn(x, t)
            x_ = (x - sigma * eps) / mu

            log_probs = []
            for likelihood, gamma in zip(self.likelihoods, self.gammas):
                err = likelihood.err(x_)
                var = likelihood.std ** 2 + (sigma) ** 2 / (mu**2 + sigma**2)
                log_probs.append(-(err ** 2 / var).sum() / 2)

        for log_prob in log_probs:
            s, = torch.autograd.grad(log_prob, x, retain_graph=True)
            eps = eps - sigma * s

        return eps