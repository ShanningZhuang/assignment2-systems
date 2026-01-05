"""
Sharded Optimizer Implementation

This module implements optimizer state sharding for distributed training.
Instead of each rank maintaining optimizer states for ALL parameters,
each rank only maintains optimizer states for a SUBSET of parameters (approximately 1/world_size).

Key Concept:
- Each rank's optimizer only handles a subset of model parameters
- After each optimizer step, updated parameters are broadcast to other ranks
- This reduces memory consumption per rank (especially important for optimizers like AdamW
  which maintain 2 floats per parameter for momentum and variance)

Communication Pattern:
- Use broadcast (not all-reduce) since each rank owns a different subset of parameters
- The rank that owns a parameter broadcasts its updated value to all other ranks
"""

from typing import Any, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """
    A wrapper optimizer that shards optimizer state across distributed ranks.

    Each rank only maintains optimizer state for a subset of parameters,
    reducing memory usage. After each step, parameters are synchronized
    across ranks via broadcast.
    """

    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        """
        Initialize the sharded optimizer.

        Args:
            params: Collection of parameters to be optimized (or parameter groups).
                    These parameters will be sharded across all ranks.
            optimizer_cls: The type of optimizer to wrap (e.g., torch.optim.AdamW).
            **kwargs: Keyword arguments forwarded to the optimizer_cls constructor
                      (e.g., lr, weight_decay, betas, eps).

        Implementation hints:
        1. Store the optimizer_cls and kwargs for later use when creating the wrapped optimizer
        2. Get world_size and rank from torch.distributed
        3. Initialize empty defaults dict for the super().__init__ call
        4. Call super().__init__(params, defaults={}) - this will call add_param_group
           for each parameter group
        5. After super().__init__, create the wrapped optimizer with only the parameters
           assigned to this rank
        6. Consider how to track which parameters are assigned to which rank
        7. Consider how to handle tied weights (same parameter appearing multiple times)
        """
        
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        self._all_params = []
        self._my_params = []
        self._param_to_rank = []
        super().__init__(params, defaults={})
        
        self._wrapped_optimizer = optimizer_cls(self._my_params, **kwargs)

    def step(self, closure=None, **kwargs):
        """
        Perform a single optimization step and synchronize parameters.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional).
            **kwargs: Additional keyword arguments passed to the wrapped optimizer's step.

        Returns:
            The loss returned by the closure (if provided).

        Implementation hints:
        1. Call the wrapped optimizer's step() method with closure and kwargs
        2. After the step, synchronize parameters across ranks:
           - Each rank broadcasts its assigned parameters to all other ranks
           - Use dist.broadcast() with the appropriate source rank
        3. Consider efficiency: can you overlap broadcasts? Use async operations?
        4. Handle tied weights carefully (don't broadcast the same parameter twice)
        """
        # TODO: Call wrapped optimizer's step

        # TODO: Synchronize parameters across ranks using broadcast
        # For each parameter, the rank that owns it should broadcast to all other ranks

        raise NotImplementedError("Implement step")

    def add_param_group(self, param_group: dict[str, Any]):
        """
        Add a parameter group to the sharded optimizer.

        This method is called:
        1. During construction by the super().__init__ for initial parameters
        2. Potentially during training (e.g., for gradually unfreezing layers)

        Args:
            param_group: A dict containing 'params' key with parameters to add,
                        plus any group-specific hyperparameters.

        Implementation hints:
        1. Assign parameters to ranks in a deterministic way across all ranks
           - A simple approach: assign parameter i to rank (i % world_size)
           - Consider load balancing based on parameter sizes
        2. Handle tied weights: same parameter may appear multiple times
           - Use a set to track which parameters have been assigned
           - Assign each unique parameter to exactly one rank
        3. Track which parameters this rank owns vs. which it doesn't
        4. Call super().add_param_group() with appropriate parameters
        5. If the wrapped optimizer already exists, also add to it

        Important considerations:
        - All ranks must agree on the assignment (deterministic algorithm)
        - Each unique parameter should be assigned to exactly one rank
        - Parameters that don't require gradients can be skipped
        """
        
        # TODO: Extract parameters from param_group
        
        # TODO: Assign parameters to ranks (deterministic across all ranks)
        # Consider: round-robin, or balanced by parameter numel()

        # TODO: Track which parameters this rank is responsible for

        # TODO: Call super().add_param_group()

        # TODO: If wrapped optimizer exists, update it as well

        raise NotImplementedError("Implement add_param_group")

    def zero_grad(self, set_to_none: bool = True):
        """
        Clear gradients of all parameters.

        Implementation hint:
        - You may need to zero gradients for ALL parameters (not just this rank's),
          since backward pass computes gradients for all parameters on all ranks
        """
        # TODO: Zero gradients for all parameters
        raise NotImplementedError("Implement zero_grad")

    @property
    def state(self):
        """
        Return the optimizer state.

        Implementation hint:
        - Return the wrapped optimizer's state
        """
        raise NotImplementedError("Implement state property")


def get_sharded_optimizer(params, optimizer_cls: Type[Optimizer], **kwargs) -> Optimizer:
    """
    Factory function to create a ShardedOptimizer.

    This is the entry point called by the adapter.
    """
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
