from .optimizer import Optimizer


class Gd(Optimizer):
    """Gradient descent optimizer."""

    def __init__(self, eta):
        """Initialize the optimizer.

        Args:
            eta (float): Learning rate.
        """
        super().__init__(eta)
        

    def step(self, params, grads):
        """Update the parameters. 
        This could be done inplace or not deppending on the rest of the code.
        Therefore you may or may not need to return the updated parameters and may or may not receive params and grads as arguments. 
        """
        pass

