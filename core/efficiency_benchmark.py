import time
import torch
import psutil
from core.models.autoencoder import EnhancementAutoencoder

def benchmark_model(device='cpu'):
    model = EnhancementAutoencoder().to(device)
    model.eval()
    
    # Dummy input representing a 224x224 slice
    dummy_input = torch.randn(1, 1, 224, 224).to(device)
    
    # Warmup
    for _ in range(5):
        _ = model(dummy_input)
        
    # Latency & Throughput
    iterations = 50
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    latency_ms = (total_time / iterations) * 1000
    throughput = iterations / total_time
    
    # Parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Memory
    mem_mb = 0
    if device == 'cuda':
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        # Process memory as proxy for CPU
        process = psutil.Process()
        mem_mb = process.memory_info().rss / (1024 * 1024)
        
    return {
        "Latency (ms)": round(latency_ms, 2),
        "Throughput (FPS)": round(throughput, 2),
        "Parameters": param_count,
        "Peak Memory (MB)": round(mem_mb, 2),
        "Device": device.upper()
    }
