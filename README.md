# cuBLAS peformance tests

Tests the execution times of `cuBlasGemmEx` with multiple precisions and math modes (to test the precision and performance when tensor cores are used or not).

```
    std::vector<int> sizesS = {32, 512, 1024, 2048, 4096};
    gemm_test<float>(mathMode, sizesS);
```

When `mathMode` is `0`, it uses `CUBLAS_COMPUTE_*F` (normal mode, uses tensor core when possible), while `1` means pedantic computation (tensor cores and other optimizations are disabled).

Another two math modes can be defined for single precision: `CUBLAS_COMPUTE_32F_FAST_TF32`, `CUBLAS_COMPUTE_32F_FAST_16F`, `CUBLAS_COMPUTE_32F_FAST_16BF` with `mathMode` 3, 4, 5, respectively.

`A5000` results:
```
PREC    SIZE    MATH MODE   MSE         GFLOPS
Half precision
-----------------------------------------------
half	32	    NORMAL	    4.531e-07	5.92593
half	512	    NORMAL	    2.91318e-07	25700.4
half	1024	NORMAL	    1.54376e-06	59918.6
half	2048	NORMAL	    9.84018e-07	84142.5
half	4096	NORMAL	    3.51929e-07	96922.5

half	32	    PEDANTIC	0	        4.35374
half	512	    PEDANTIC	0	        8828.26
half	1024	PEDANTIC	0	        20846.4
half	2048	PEDANTIC	0	        26329.6
half	4096	PEDANTIC	0	        27219.7

Single precision
-----------------------------------------------
single	32	    NORMAL	    0	        5.56522
single	512	    NORMAL	    0	        9429.64
single	1024	NORMAL	    0	        14141.3
single	2048	NORMAL	    0	        19939.6
single	4096	NORMAL	    0	        20479.1

single	32	    PEDANTIC	0	        5.76576
single	512	    PEDANTIC	0	        9393.74
single	1024	PEDANTIC	0	        14149.9
single	2048	PEDANTIC	0	        19930.5
single	4096	PEDANTIC	0	        20572

single	32	    FAST_TF32	0	        5.76577
single	512	    FAST_TF32	8.69421e-09	16697.1
single	1024	FAST_TF32	6.29232e-09	36868.9
single	2048	FAST_TF32	2.75609e-09	45319.3
single	4096	FAST_TF32	9.62924e-10	50776.6

single	32	    FAST_16F	0	        5.66372
single	512	    FAST_16F	7.85101e-09	17712.4
single	1024	FAST_16F	4.35854e-09	43416.5
single	2048	FAST_16F	1.29975e-09	68618.5
single	4096	FAST_16F	1.1082e-09	74091.8

single	32	    FAST_16BF	0	        5.98131
single	512	    FAST_16BF	5.19268e-07	12483
single	1024	FAST_16BF	2.13108e-07	42799
single	2048	FAST_16BF	9.73479e-08	68092.8
single	4096	FAST_16BF	6.62734e-08	74297.4

Double precision
-----------------------------------------------
double	2048	NORMAL	    0	        417.635

double	2048	PEDANTIC	0	        416.559
```
