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


class DDPBucketed(nn.Module):
    """
    DDP implementation with gradient bucketing and overlap.

    This implementation:
    1. Groups parameters into buckets (each bucket <= bucket_size_mb)
    2. Allocates buckets in REVERSE order of model.parameters() since gradients
       become ready in approximately that order during backward pass
    3. Uses gradient hooks to detect when gradients are ready
    4. Launches async all-reduce on a bucket when ALL its gradients are ready
    5. Overlaps communication with backward computation

    Design:
    - Each bucket has a pre-allocated buffer for flattened gradients
    - Hooks copy gradients into bucket buffers and track readiness
    - When bucket is full (all grads ready), launch async all-reduce
    - finish_gradient_synchronization() waits for all handles and copies back

    Args:
        module: The model to wrap
        bucket_size_mb: Maximum size of each bucket in megabytes.
                       If 0 or very large, all parameters go in one bucket.
    """

    def __init__(self, module: torch.nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.world_size = dist.get_world_size()

        # Check backend - NCCL supports AVG, Gloo does not
        backend = dist.get_backend()
        self.use_avg_op = (backend == "nccl")

        # Broadcast parameters from rank 0
        self._broadcast_parameters()

        # Create buckets and register hooks
        self._create_buckets()
        self._register_hooks()

        # Storage for async handles during backward
        self.bucket_handles = []  # List of (handle, bucket_id)

    def _broadcast_parameters(self):
        """Broadcast parameters from rank 0 to all other ranks."""
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def _create_buckets(self):
        """
        Create buckets by grouping parameters in REVERSE order.

        Gradients become ready in reverse order during backward pass,
        so we want early buckets (lower indices) to contain parameters
        from later layers (which get their gradients first).

        Bucket structure:
        - self.buckets: List of bucket info dicts
        - self.param_to_bucket: Maps param id -> (bucket_id, offset_in_bucket)
        """
        bucket_size_bytes = self.bucket_size_mb * 1024 * 1024

        # Collect parameters that require gradients, in REVERSE order
        params_reversed = []
        for param in self.module.parameters():
            if param.requires_grad:
                params_reversed.append(param)
        params_reversed = list(reversed(params_reversed))

        # Group parameters into buckets
        self.buckets = []  # List of bucket dicts
        self.param_to_bucket = {}  # param_id -> (bucket_id, offset_in_bucket)

        current_bucket_params = []
        current_bucket_size = 0

        for param in params_reversed:
            param_size = param.numel() * param.element_size()

            # Check if adding this param would exceed bucket size
            # (unless bucket is empty, always add at least one param)
            if current_bucket_size + param_size > bucket_size_bytes and current_bucket_params:
                # Finalize current bucket
                self._finalize_bucket(current_bucket_params)
                current_bucket_params = []
                current_bucket_size = 0

            # Add param to current bucket
            current_bucket_params.append(param)
            current_bucket_size += param_size

        # Finalize last bucket if non-empty
        if current_bucket_params:
            self._finalize_bucket(current_bucket_params)

    def _finalize_bucket(self, params):
        """Create a bucket from a list of parameters."""
        bucket_id = len(self.buckets)

        # Calculate total size and offsets
        total_numel = sum(p.numel() for p in params)

        # Determine dtype and device from first param
        dtype = params[0].dtype
        device = params[0].device

        # Create bucket buffer (pre-allocated for efficiency)
        bucket_buffer = torch.zeros(total_numel, dtype=dtype, device=device)

        # Track offsets for each param in this bucket
        param_info = []  # List of (param, offset, numel)
        offset = 0
        for param in params:
            numel = param.numel()
            self.param_to_bucket[id(param)] = (bucket_id, offset, numel)
            param_info.append((param, offset, numel))
            offset += numel

        bucket = {
            'id': bucket_id,
            'params': params,
            'param_info': param_info,
            'buffer': bucket_buffer,
            'num_params': len(params),
            'grads_ready': 0,  # Counter for ready gradients
            'total_numel': total_numel,
        }
        self.buckets.append(bucket)

    def _register_hooks(self):
        """Register gradient hooks on all parameters that require gradients."""
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._make_hook(param))

    def _make_hook(self, param):
        """
        Create a gradient hook for the given parameter.

        The hook:
        1. Copies gradient to bucket buffer
        2. Increments bucket's ready counter
        3. If bucket is full, launches async all-reduce
        """
        param_id = id(param)

        def hook(grad):
            if grad is not None and param_id in self.param_to_bucket:
                bucket_id, offset, numel = self.param_to_bucket[param_id]
                bucket = self.buckets[bucket_id]

                # Ensure gradient is contiguous
                if not grad.is_contiguous():
                    grad = grad.contiguous()

                # Copy gradient to bucket buffer
                bucket['buffer'][offset:offset + numel].copy_(grad.view(-1))

                # Increment ready counter
                bucket['grads_ready'] += 1

                # Check if bucket is ready for all-reduce
                if bucket['grads_ready'] == bucket['num_params']:
                    with nvtx.range(f"launch_bucket_allreduce_{bucket_id}"):
                        # Use AVG for NCCL (more efficient), SUM for Gloo
                        reduce_op = dist.ReduceOp.AVG if self.use_avg_op else dist.ReduceOp.SUM
                        handle = dist.all_reduce(
                            bucket['buffer'],
                            op=reduce_op,
                            async_op=True
                        )
                        self.bucket_handles.append((handle, bucket_id))

            return grad

        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        Wait for all async all-reduce operations to complete and
        copy averaged gradients back to parameter .grad tensors.
        """
        with nvtx.range("finish_bucketed_gradient_sync"):
            # Wait for all bucket all-reduces to complete
            for handle, bucket_id in self.bucket_handles:
                with nvtx.range(f"wait_bucket_{bucket_id}"):
                    handle.wait()

            # Copy averaged gradients back to parameters
            # Track processed params to handle tied weights
            processed_params = set()

            for bucket in self.buckets:
                # Only divide if we used SUM (Gloo backend)
                # NCCL uses AVG which already averages
                if not self.use_avg_op:
                    bucket['buffer'].div_(self.world_size)

                # Copy back to each parameter's gradient
                for param, offset, numel in bucket['param_info']:
                    if id(param) not in processed_params:
                        if param.grad is not None:
                            param.grad.data.copy_(
                                bucket['buffer'][offset:offset + numel].view(param.grad.shape)
                            )
                        processed_params.add(id(param))

            # Clear handles for next iteration
            self.bucket_handles.clear()

    def reset_buckets(self):
        """Reset bucket state for the next iteration."""
        for bucket in self.buckets:
            bucket['grads_ready'] = 0
            bucket['buffer'].zero_()
