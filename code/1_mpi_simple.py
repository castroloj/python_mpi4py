from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data = 42
    for i in range(1, size):
        comm.send(data+i, dest=i)
        
    for i in range(1, size):
        result = comm.recv(source=i)
        print(f"Rank 0 received result: {result} from rank {i}")
else:
    received = comm.recv(source=0)
    print(f"Rank {rank} received value: {received}")
    computed = received * 2
    comm.send(computed, dest=0)

MPI.Finalize()