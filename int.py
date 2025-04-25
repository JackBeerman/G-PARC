import torch
import torch.nn as nn

class Integrator(nn.Module):
    """
    Integrator without PoissonBlock and data-driven integrators.

    Args:
        clip (bool): Whether to clip the state or velocity variable before each integration step.
        num_int (nn.Module): Numerical integrator.
        **kwargs: Additional arguments for nn.Module initialization.
    """

    def __init__(
        self,
        clip: bool,
        num_int: nn.Module,
        **kwargs,
    ):
        super(Integrator, self).__init__(**kwargs)
        self.clip = clip
        self.numerical_integrator = num_int

    def forward(self, f, ic, t0, t1):
        """
        Forward pass of Integrator. Clips the current state if necessary and applies the numerical integrator.

        Args:
            f (callable): Callable that returns the time derivative.
            ic (torch.Tensor): 4D tensor of shape (batch_size, channels, y, x), representing the initial condition.
            t0 (float): Starting time.
            t1 (torch.Tensor): 1D tensor of time points to integrate to.

        Returns:
            torch.Tensor: 5D tensor of shape (timesteps, batch_size, channels, y, x) with the predicted states at each time point in t1.
        """
        all_time = torch.cat([t0.unsqueeze(0), t1])
        res = []
        current = ic

        for ts in range(1, all_time.size(0)):
            if self.clip:
                current = torch.clamp(current, 0.0, 1.0)

            # Apply numerical integrator
            current, update = self.numerical_integrator(
                f,
                all_time[ts - 1],
                current,
                all_time[ts] - all_time[ts - 1]
            )

            res.append(current.unsqueeze(0))

        res = torch.cat(res, dim=0)
        return res
