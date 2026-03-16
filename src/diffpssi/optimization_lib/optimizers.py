"""This file contains the custom optimizers.

It is used for the optimization of the power system parameters.
"""

# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments, too-many-locals, invalid-name
import torch


class CustomBFGSREALOptimizer(torch.optim.Optimizer):
    """
    Custom optimizer for power system parameter optimization using BFGS.

    This optimizer reduces the learning rate each step,
    automatically chooses a reasonable learning rate,
    and enforces a maximum relative step size for parameter optimization.
    The optimizer is based on the BFGS algorithm.
    The line search is omitted, as it is not possible to do for batched simulations.
    """

    # Init Method:
    def __init__(self, params, max_step=0.1, decay=0.9, verbose=True):
        """
        Initialize the CustomBFGSREALOptimizer.

        Args:
            params (iterable): Parameters to optimize.
            max_step (float or list, optional): Maximum relative step size for each parameter.
            decay (float, optional): Decay factor for the maximum step size.
            verbose (bool, optional): If True, print step size changes.
        """
        self.decay = decay
        # Max step can be a list of values (one for each parameter), or a single value.
        # For most power system parameters, a single value equal to 0.1 is a good choice.
        try:
            max_steps = [max_step[i] * abs(p.data.real) for i, p in enumerate(params)]
        except TypeError:
            max_steps = [max_step * abs(p.data.real) for p in params]

        self.max_step = max_step
        self.decay = decay
        self.verbose = verbose

        defaults = {"max_steps": max_steps}
        # super(CustomBFGSREALOptimizer, self).__init__(params, defaults=defaults)
        super().__init__(params, defaults=defaults)

        # initialize the hessian as the identity matrix
        self.hesse = torch.eye(len(params), dtype=torch.float64).expand(
            len(params[0]), -1, -1
        )

        # store the last parameters in order to approximate the second derivative
        self.last_grads = None
        self.first_step = True
        self.last_sk = None

    def decrease_step_size(self):
        """
        Decrease the maximum step size by multiplying it with the decay factor.

        This should always be done after the loss did not decrease,
        as it indicates we are close to the optimum and optimizing too aggressively.

        Returns:
            None
        """
        self.max_step *= self.decay
        if self.verbose:
            print(f"Decreasing max. step size to {round(self.max_step, 4)}")

    def step(self, closure=None):
        """
        Perform a single optimization step.

        The step size is chosen by the maximum step size and the decay factor.
        The optimization is done similar to the BFGS algorithm, but without the line search.

        Args:
            **kwargs: Additional keyword arguments (not used).

        Returns:
            None
        """
        # 1.: get the gradients from the "Backpropagation" step
        grads_list = []
        for p in self.param_groups[0]["params"]:
            if p.grad is None:
                raise RuntimeError(
                    "Gradient for an optimizable parameter is None. "
                    "Ensure that `loss.backward()` has been called and that all parameters "
                    "in `params_optimizable` actually affect the loss."
                )
            grads_list.append(p.grad.real)
        grads = torch.stack(grads_list, dim=1)

        if self.first_step:
            # ensure sufficiently large first step by initializing the hessian
            # in a way so that the first step always equals the maximum step size.
            # This is only done for the first step.
            self.first_step = False
            directions = torch.stack(
                [p.data.real * self.max_step for p in self.param_groups[0]["params"]],
                dim=1,
            )
            # directions divided by gradients multiplied with the hessian to equal the shape
            self.hesse = self.hesse * directions / torch.abs(grads)

        # 2.: get the direction of the step by multiplying the hessian with the gradients
        direction_p = torch.matmul(-self.hesse, grads)

        # Ensure that no direction is larger than the maximum step size
        # by setting a multiplier a that reduces the step size of the direction
        max_steps = torch.stack(
            [
                self.max_step * abs(p.data.real) / abs(direction_p[:, idx])
                for idx, p in enumerate(self.param_groups[0]["params"])
            ]
        )
        a = torch.min(torch.tensor(1), torch.min(torch.abs(max_steps), dim=0).values)

        # 3. get sk and update parameters
        sk = a.unsqueeze(-1) * direction_p
        for group in self.param_groups:
            for p_idx, p in enumerate(group["params"]):
                p.data += sk[:, p_idx]

        # 4. get yk to update Hesse
        if self.last_grads is None:
            # Initial gradients, no update of Hessian is possible yet
            self.last_grads = grads
            self.last_sk = sk
        else:
            yk = grads - self.last_grads

            # dims yk and sk (3, 1, 1000) -> (1000, 3, 1)
            ykT = torch.transpose(yk, -1, -2)
            last_skT = torch.transpose(self.last_sk, -1, -2)

            rho = 1 / torch.matmul(ykT, self.last_sk)

            # 5. update Hesse
            left = torch.eye(
                len(self.param_groups[0]["params"]), dtype=torch.float64
            ).expand(
                len(self.param_groups[0]["params"][0]), -1, -1
            ) - rho * torch.matmul(
                self.last_sk, ykT
            )
            right = torch.eye(
                len(self.param_groups[0]["params"]), dtype=torch.float64
            ).expand(
                len(self.param_groups[0]["params"][0]), -1, -1
            ) - rho * torch.matmul(
                yk, last_skT
            )
            end = rho * torch.matmul(self.last_sk, last_skT)

            # calculate the new hessian
            self.hesse = torch.matmul(left, torch.matmul(self.hesse, right)) + end

            # update the last gradients and sk
            self.last_grads = grads
            self.last_sk = sk
