import torch
def total_current_mem():
    """Sum current memory across all GPUs (in MB)."""
    total = 0
    for i in range(torch.cuda.device_count()):
            total += torch.cuda.memory_allocated(i)
    return total / 1e6


def total_peak_mem():
    """Sum peak memory across all GPUs (in MB)."""
    total = 0
    for i in range(torch.cuda.device_count()):
            total += torch.cuda.max_memory_allocated(i)
    return total / 1e6