import torch
import time

# Matrix size (choose a large matrix to see the difference)
MATRIX_SIZE = 10000

# Create random matrices
cpu_matrix1 = torch.randn(MATRIX_SIZE, MATRIX_SIZE)
cpu_matrix2 = torch.randn(MATRIX_SIZE, MATRIX_SIZE)

# GPU matrices (if CUDA is available)
if torch.cuda.is_available():
    gpu_matrix1 = cpu_matrix1.to('cuda')
    gpu_matrix2 = cpu_matrix2.to('cuda')
else:
    print("CUDA is not available. Exiting...")
    exit()

# CPU computation
start_time = time.time()
cpu_result = torch.matmul(cpu_matrix1, cpu_matrix2)
cpu_time = time.time() - start_time
print(f"CPU computation took {cpu_time:.6f} seconds")

# GPU computation (Warm-up the GPU)
torch.matmul(gpu_matrix1, gpu_matrix2)  # First operation to initialize GPU (optional)

# GPU computation
start_time = time.time()
gpu_result = torch.matmul(gpu_matrix1, gpu_matrix2)
torch.cuda.synchronize()  # Wait for GPU operations to finish
gpu_time = time.time() - start_time
print(f"GPU computation took {gpu_time:.6f} seconds")

# Check that results are almost identical
if torch.allclose(cpu_result, gpu_result.cpu(), atol=1e-6):
    print("The results are identical within tolerance.")

# Print the speedup
speedup = cpu_time / gpu_time
print(f"GPU is {speedup:.2f} times faster than CPU for this operation.")

