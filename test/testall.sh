mpirun -np 16 ./test_1D_partition.x
mpirun -np 16 ./test_batch_acc.x
mpirun -np 16 ./test_nonblk_acc.x
mpirun -np 16 ./test_Symmetrize.x
mpirun -np 16 ./test_task_queue.x
mpirun -np 4  ./test_complex.x