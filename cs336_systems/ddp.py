import torch
import torch.nn as nn
import torch.distributed as dist

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
        def hook(grad):
            if grad is not None:
                handle = dist.all_reduce(grad.data, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append((handle, param))
            return grad
        return hook
    
    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)
    
    def finish_gradient_synchronization(self):
        """Wait for all async all-reduce operations to complete"""
        divided_params = set()
        
        for handle, param in reversed(self.handles):
            handle.wait()
            # Only divide once per unique parameter (handles tied weights)
            if param.grad is not None and id(param) not in divided_params:
                param.grad.div_(dist.get_world_size())
                divided_params.add(id(param))
        self.handles.clear()
            
        
    