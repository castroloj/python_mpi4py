from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = 42
    comm.send(data, dest=1)
    result = comm.recv(source=1)
    print(f"Rank 0 received result: {result}")
elif rank == 1:
    received = comm.recv(source=0)
    print(f"Rank 1 received value: {received}")
    computed = received * 2
    comm.send(computed, dest=0)

MPI.Finalize()