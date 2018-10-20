# Buzz Matrix
Copyright (c) 2018 Georgia Institute of Technology

Buzz Matrix is yet another tiny PGAS (partitioned global address space) framework. Its original purpose is to replace [Global Arrays](http://hpc.pnl.gov/globalarrays/) in [GTFock](https://github.com/gtfock-chem/gtfock). It is written in C and uses only MPI. 

Supports:

* Matrix only (row-major style)
* Data type: int, double
* Operations: get, put, accumulate, fill, symmetrize

## Compiling and Linking
To compile Buzz Matrix, you just need to run `make` in this directory. Buzz Matrix will be compiled into `libBuzzMatrix.a`. 

To use Buzz Matrix, you need to add this directory to the C include file path and include `Buzz_Matrix.h` in the source file. Link `libBuzzMatrix.a` to use the compiled Buzz Matrix.

Test passed: Stampede2 supercomputer, Intel compiler 17.0.4 + Intel MPI 17.0.3, 64 nodes * 4 MPI processes per node.

## Brief Manual

Each C file has a corresponding header file. Please refer to header files to see detail parameters of each function. 


Create a Buzz Matrix object: `Buzz_createBuzzMatrix(Buzz_Matrix_Obj, ...)`
Destroy a Buzz Matrix object: `Buzz_destroyBuzzMatrix(Buzz_Matrix_Obj)`


Get a block: two methods. `Buzz_getBlock(Buzz_Matrix_Obj, ..., 1)` is a blocking operation that gets the target block and then return. If you have several get requests, you can use the batch operation mode for better performance:
```c
Buzz_startBatchGet(Buzz_Matrix_Obj);
for (int i = 0; i < nblocks; i++)
    Buzz_addGetBlockRequest(Buzz_Matrix_Obj, ...);
Buzz_execBatchGet(Buzz_Matrix_Obj);
Buzz_stopBatchGet(Buzz_Matrix_Obj);
```


Update (put or accumulate) a block: also two methods. `Buzz_putBlock(Buzz_Matrix_Obj, ...)` / `Buzz_accumulateBlock(Buzz_Matrix_Obj, ...)` is a blocking operation that puts / accumulates a local block to the target block and then return. The batch operation mode for update is almost the same as the batch operation mode for get:
```c
Buzz_startBatchUpdate(Buzz_Matrix_Obj);
for (int i = 0; i < nblocks; i++)
    Buzz_addPutBlockRequest(Buzz_Matrix_Obj, ...);
//or : Buzz_addAccumulateBlockRequest(Buzz_Matrix_Obj, ...);
Buzz_execBatchUpdate(Buzz_Matrix_Obj);
Buzz_stopBatchUpdate(Buzz_Matrix_Obj);
```
**NOTICE:** Buzz Matrix guarantee the element-wise atomicity for accumulation. For put operations, Buzz Matrix does not guarantee the actual behavior and correctness when a block is updated by several processes at the same time. Also, try to avoid using Buzz Matrix in a way that some processes are updating some blocks while other processes are reading some blocks. 


Synchronization (barrier): `Buzz_Sync(Buzz_Matrix_Obj)`.


Symmetrizing a Buzz Matrix: `Buzz_Symmetrize(Buzz_Matrix_Obj)`.


Fill a Buzz Matrix with a given value: `Buzz_fillBuzzMatrix(Buzz_Matrix_Obj, ...)`.


## Known Issue
When compiled with Intel MPI 17.0.3 on KNL or Skylake processor and running more than 16 MPI processes on a single node, or running more than one MPI process per node on multiple nodes, a program needs to use `Buzz_Sync()` after some operations on a Buzz Matrix to avoid deadlock. Replace `Buzz_Sync()` with `MPI_Barrier()` will lead to deadlock, reason unknown. 