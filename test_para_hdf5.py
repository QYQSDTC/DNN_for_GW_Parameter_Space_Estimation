from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Rank {rank}/{size} says hello")

filename = "test_parallel.h5"
with h5py.File(filename, "w", driver="mpio", comm=comm) as f:
    dset = f.create_dataset("data", (size,), dtype="i")
    dset[rank] = rank

if rank == 0:
    print("Data written to 'test_parallel.h5'")
