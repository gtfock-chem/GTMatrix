# GTMatrix
Copyright (c) 2018-2019 Edmond Group at Georgia Tech

Authors:
Hua Huang ([huangh223@gatch.edu](mailto:huangh223@gatch.edu))
Edmond Chow ([echow@cc.gatech.edu](mailto:echow@cc.gatech.edu))

GTMatrix is a portable and lightweight PGAS (partitioned global address space) framework. It is written in C + MPI-3 and provides C/C++ interface.

Supports:

* Matrix only (row-major style)
* Data type: int, double
* Operations: get, put, accumulate, fill, symmetrize

## Compiling and Linking
To compile GTMatrix, you just need to run `make` in this directory. Please modify the `Makefile` to change the compiler if your are not using Intel MPI. GTMatrix will be compiled into `libGTMatrix.a`. 

To use GTMatrix, you need to add this directory to the C include file path and include `GTMatrix.h` in the source file. Link `libGTMatrix.a` to use the compiled GTMatrix.

Test passed: Stampede2 supercomputer, Intel compiler 17.0.4 + Intel MPI 17.0.3, 64 nodes * 4 MPI processes per node.

## Brief Manual

Each C file has a corresponding header file. Please refer to header files to see detail parameters of each function. 


Create a GTMatrix object: `GTM_createGTMatrix(GTMatrix_t, ...)`
Destroy a GTMatrix object: `GTM_destroyGTMatrix(GTMatrix_t)`


Get a block: two methods. `GTM_getBlock(GTMatrix_t, ..., 1)` is a blocking operation that gets the target block and then return. If you have several get requests, you can use the batch operation mode for better performance:
```c
GTM_startBatchGet(GTMatrix_t);
for (int i = 0; i < nblocks; i++)
    GTM_addGetBlockRequest(GTMatrix_t, ...);
GTM_execBatchGet(GTMatrix_t);
GTM_stopBatchGet(GTMatrix_t);
```


Update (put or accumulate) a block: also two methods. `GTM_putBlock(GTMatrix_t, ...)` / `GTM_accumulateBlock(GTMatrix_t, ...)` is a blocking operation that puts / accumulates a local block to the target block and then return. The batch operation mode for update is almost the same as the batch operation mode for get:
```c
GTM_startBatchUpdate(GTMatrix_t);
for (int i = 0; i < nblocks; i++)
    GTM_addPutBlockRequest(GTMatrix_t, ...);
//or : GTM_addAccumulateBlockRequest(GTMatrix_t, ...);
GTM_execBatchUpdate(GTMatrix_t);
GTM_stopBatchUpdate(GTMatrix_t);
```
**NOTICE:** GTMatrix guarantee the element-wise atomicity for accumulation. For put operations, GTMatrix does not guarantee the actual behavior and correctness when a block is updated by several processes at the same time. Also, try to avoid using GTMatrix in a way that some processes are updating some blocks while other processes are reading some blocks. 


Synchronization (barrier): `GTM_Sync(GTMatrix_t)`.


Symmetrizing a GTMatrix: `GTM_symmetrizeGTMatrix(GTMatrix_t)`.


Fill a GTMatrix with a given value: `GTM_fillGTMatrix(GTMatrix_t, ...)`.


## Known Issue
When compiled with Intel MPI 17.0.3 on KNL or Skylake processor and running more than 16 MPI processes on a single node, or running more than one MPI process per node on multiple nodes, a program needs to use `GTM_Sync()` after some operations on a GTMatrix to avoid deadlock. Replace `GTM_Sync()` with `MPI_Barrier()` will lead to deadlock, reason unknown. 