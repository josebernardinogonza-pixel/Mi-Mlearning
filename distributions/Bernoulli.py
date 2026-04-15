import torch
from torch.distributions import Bernoulli as Bernoulli_Dist
from ..utils import *
from .distribution_utils import *

class Bernoulli:
    """
    Bernoulli distribution class.

    The Bernoulli distribution is a discrete probability distribution that models the probability of a single
    event occurring (success) or not occurring (failure).

    Args:
        stabilize (bool): Whether to use softplus to stabilize the scale parameter.
        response_fn (str): Response function for the location parameter.
    """
    def __init__(self, 
                 stabilize: bool = "none",
                 response_fn: str = "sigmoid"
                 ):
        self.n_params = 1
        self.parameter_names = ["probs"]
        self.stabilize = stabilize
        self.response_fn = response_fn

    def initialize(self, target: torch.Tensor):
        """
        Initialize the distribution parameters.

        Args:
            target (torch.Tensor): Target variable.

        Returns:
            torch.Tensor: Initialized parameters.
        """
        probs = torch.mean(target.float())
        if self.response_fn == "sigmoid":
            res = torch.log(probs / (1 - probs))
        else:
            res = probs
        return res

    def log_prob(self, 
                 target: torch.Tensor, 
                 params: torch.Tensor
                 ):
        """
        Log-probability function.

        Args:
            target (torch.Tensor): Target variable.
            params (torch.Tensor): Distribution parameters.

        Returns:
            torch.Tensor: Log-probability.
        """
        probs = self.draw_samples(params)
        dist = Bernoulli_Dist(probs=probs)
        log_prob = dist.log_prob(target.float())
        return log_prob

    def draw_samples(self, params: torch.Tensor):
        """
        Draw samples from the distribution.

        Args:
            params (torch.Tensor): Distribution parameters.

        Returns:
            torch.Tensor: Samples.
        """
        if self.response_fn == "sigmoid":
            probs = torch.sigmoid(params)
        else:
            probs = params
        return probs
