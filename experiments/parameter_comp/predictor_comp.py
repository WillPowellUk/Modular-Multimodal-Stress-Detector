from src.ml_pipeline.models.late_fusion_models.hard_voting import *
from src.ml_pipeline.models.late_fusion_models.soft_voting import *
from src.ml_pipeline.models.late_fusion_models.kalman import *
import torch
import time
import numpy as np

embed_dim = 16
n_branches = [3, 5, 8]
for m in n_branches:
    branch_keys = [f'{i}' for i in range(m)]

    # Instantiate the models with dummy arguments
    stacked_modular_pool = StackedModularPool(embed_dim=embed_dim, hidden_dim=32, output_dim=2, dropout=0.5, pool_type="avg")
    modular_pool = ModularPool(embed_dim=embed_dim, output_dim=2, dropout=0.5, branch_keys=branch_keys, pool_type="avg")
    modular_weighted_pool = ModularWeightedPool(embed_dim=embed_dim, output_dim=2, dropout=0.5, branch_keys=branch_keys, pool_type="avg")
    hard_voting = ModularHardVoting(embed_dim=embed_dim, output_dim=2, dropout=0.5, branch_keys=branch_keys)
    kalman_modular_pool = ModularPool(embed_dim=embed_dim, output_dim=2, dropout=0.5, branch_keys=branch_keys, pool_type="avg", return_branch_outputs=True)
    kalman_filter = KalmanFilter(num_classes=2, num_branches=len(branch_keys), device='cpu')

    models = {
        'HardVoting': hard_voting,
        'StackedModularPool': stacked_modular_pool,
        'ModularPool': modular_pool,
        'ModularWeightedPool': modular_weighted_pool,
        'KalmanModularPool+KalmanFilter': (kalman_modular_pool, kalman_filter)
    }

    # Define a function to count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n--- For {m} branches ---")
    for name, model in models.items():
        if isinstance(model, tuple):
            # Count parameters for both kalman_modular_pool and kalman_filter
            total_params = sum(count_parameters(m) for m in model)
            print(f"Number of parameters in {name}: {total_params}")
        else:
            print(f"Number of parameters in {name}: {count_parameters(model)}")

    # Define a dummy input tensor of (bath size, sequence length, embed_dim)
    x_cpu = {key: torch.randn(1, 1, embed_dim) for key in branch_keys}
    x_cuda = {key: tensor.cuda() for key, tensor in x_cpu.items()} if torch.cuda.is_available() else None

    for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
        print(f"\nInference timings on {device.upper()}:")
        for name, model in models.items():
            if device == 'cuda':
                if isinstance(model, tuple):
                    for m in model:
                        m.cuda()
                else:
                    model.cuda()
                # Warm-up runs
                for _ in range(10):
                    with torch.no_grad():
                        if isinstance(model, tuple):
                            model[1].device = device
                            out_mod_pool = model[0](x_cuda)
                            output = model[1](out_mod_pool)
                        else:
                            output = model(x_cuda)
                torch.cuda.synchronize()
            else:
                if isinstance(model, tuple):
                    for m in model:
                        m.cpu()
                else:
                    model.cpu()

            times = []
            for _ in range(100):
                if device == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                with torch.no_grad():
                    if isinstance(model, tuple):
                        model[1].device = device
                        out_mod_pool = model[0](x_cpu) if device == 'cpu' else model[0](x_cuda)
                        output = model[1](out_mod_pool)
                    else:
                        output = model(x_cpu) if device == 'cpu' else model(x_cuda)
                if device == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = np.mean(times)
            std_time = np.std(times)
            print(f"{name}: Avg Time = {avg_time*1000:.4f} ms, Std Dev = {std_time*1000:.4f} ms")
