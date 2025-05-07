from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ensure size divides data evenly
assert size > 1, "Run with multiple processes"

N = 25_600_000

# --- Standard communication with Python lists ---
if rank == 0:
    data_list = list(range(N))
    chunk_size = len(data_list) // size
    chunks_list = [data_list[i*chunk_size:(i+1)*chunk_size] for i in range(size)]
else:
    chunks_list = None

# Synchronize
comm.Barrier()
start = time.time()

# Scatter and compute
local_data_list = comm.scatter(chunks_list, root=0)
local_result_list = [x * x for x in local_data_list]

# Gather
gathered_list = comm.gather(local_result_list, root=0)

end = time.time()
if rank == 0:
    print(f"[List] Total time: {end - start:.4f} seconds")

# --- Buffered communication with NumPy arrays ---
comm.Barrier()

data_array = None
if rank == 0:
    data_array = np.arange(N, dtype=np.int32)
    chunk_size = len(data_array) // size
    chunks_array = np.array_split(data_array, size)
else:
    chunks_array = None

local_data_array = np.empty(N // size, dtype=np.int32)

start = time.time()

# Scatter (buffered)
comm.Scatter([data_array, MPI.INT], local_data_array, root=0)

# Compute
local_result_array = local_data_array * local_data_array

# Gather (buffered)
gathered_array = None
if rank == 0:
    gathered_array = np.empty_like(data_array)

comm.Gather(local_result_array, gathered_array, root=0)

end = time.time()

if rank == 0:
    print(f"[NumPy] Total time: {end - start:.4f} seconds")
