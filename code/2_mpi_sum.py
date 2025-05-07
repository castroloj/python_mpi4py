from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Step 1: Create data only on root
if rank == 0:
    print(f'This has {size} processes')
    data = list(range(1000))
    chunks = [data[i::size] for i in range(size)]
else:
    chunks = None

# Scatter chunks
local_data = comm.scatter(chunks, root=0)

print(f"Rank: {rank} received {local_data}")

# Step 2: Local sum
local_sum = sum(local_data)

# Step 3: Reduce to total sum
total = comm.reduce(local_sum, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Total sum is: {total}")

MPI.Finalize()