import torch
import torch.nn as nn
import torch.distributed as dist
import torch.cuda.nvtx as nvtx

# define a class that like module that will be called by each process

class DDPIndividualParameters(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        
        self.handles = []
        
        self._broadcast_parameters()
        
        self._register_hooks()
        
    def _broadcast_parameters(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            
    def _register_hooks(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._make_hook(param))
    
    def _make_hook(self, param):
        param_idx = [0]  # Use list to allow mutation in nested function
        for i, p in enumerate(self.module.parameters()):
            if p is param:
                param_idx[0] = i
                break

        def hook(grad):
            if grad is not None:
                # Ensure gradient is contiguous for distributed communication
                with nvtx.range(f"make_grad_contiguous_param_{param_idx[0]}"):
                    if not grad.is_contiguous():
                        grad = grad.contiguous()

                # Launch async all-reduce
                with nvtx.range(f"launch_allreduce_param_{param_idx[0]}"):
                    handle = dist.all_reduce(grad.data, op=dist.ReduceOp.SUM, async_op=True)
                    self.handles.append((handle, param))
            return grad
        return hook
    
    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)
    
    def finish_gradient_synchronization(self):
        """Wait for all async all-reduce operations to complete"""
        with nvtx.range("finish_gradient_synchronization"):
            divided_params = set()

            # for handle, param in reversed(self.handles):
            for idx, (handle, param) in enumerate(self.handles):
                with nvtx.range(f"wait_allreduce_param_{idx}"):
                    handle.wait()
                    # Only divide once per unique parameter (handles tied weights)
                    if param.grad is not None and id(param) not in divided_params:
                        param.grad.div_(dist.get_world_size())
                        divided_params.add(id(param))
            self.handles.clear()


class DDPNaive(nn.Module):
    """
    Naive DDP implementation WITHOUT hooks or overlap.

    This implementation performs synchronous all-reduce AFTER the entire backward
    pass completes. There is NO overlap between communication and computation.

    Use this to compare against DDPIndividualParameters to see the benefit of
    overlapping communication with backward computation.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._broadcast_parameters()

    def _broadcast_parameters(self):
        """Broadcast parameters from rank 0 to all other ranks."""
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)

    def finish_gradient_synchronization(self):
        """
        Synchronize gradients across all ranks using a single flattened all-reduce.

        This flattens all gradients into one tensor, does a single all-reduce,
        then copies results back. Still NO overlap with computation.
        """
        with nvtx.range("naive_gradient_sync_flattened"):
            world_size = dist.get_world_size()

            # Collect all gradients that need synchronization
            grads = []
            grad_shapes = []
            grad_numels = []

            with nvtx.range("flatten_gradients"):
                for param in self.module.parameters():
                    if param.grad is not None:
                        grad = param.grad.data
                        if not grad.is_contiguous():
                            grad = grad.contiguous()
                        grads.append(grad)
                        grad_shapes.append(grad.shape)
                        grad_numels.append(grad.numel())

                # Flatten all gradients into a single tensor
                flat_grads = torch.cat([g.view(-1) for g in grads])

            # Single all-reduce for all gradients
            with nvtx.range("allreduce_flattened"):
                dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=False)

            # Average the gradients
            with nvtx.range("average_and_unflatten"):
                flat_grads.div_(world_size)

                # Copy results back to original gradient tensors
                offset = 0
                grad_idx = 0
                for param in self.module.parameters():
                    if param.grad is not None:
                        numel = grad_numels[grad_idx]
                        param.grad.data.copy_(flat_grads[offset:offset + numel].view(grad_shapes[grad_idx]))
                        offset += numel
                        grad_idx += 1

    # def finish_gradient_synchronization_individual(self):
    #     """
    #     Old implementation: Synchronize gradients one by one.
    #     Kept for reference.
    #     """
    #     with nvtx.range("naive_gradient_sync"):
    #         world_size = dist.get_world_size()
    #
    #         for idx, param in enumerate(self.module.parameters()):
    #             if param.grad is not None:
    #                 with nvtx.range(f"naive_allreduce_param_{idx}"):
    #                     if not param.grad.is_contiguous():
    #                         param.grad = param.grad.contiguous()
    #                     dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=False)
    #                     param.grad.div_(world_size)
