# Buzz_Matrix
Copyright (c) 2018 Georgia Institute of Technology

Buzz_Matrix is yet another tiny PGAS (partitioned global address space) framework. Its original purpose is to replace [Global Arrays](http://hpc.pnl.gov/globalarrays/) in [GTFock](https://github.com/gtfock-chem/gtfock). It is written in C and uses only MPI. It utilizes MPI-3 features to obtain better performance. 

Supports:

* Matrix only (row-major style)
* Data type: int, double
* Operations: get, put, accumulate, fill, symmetrize