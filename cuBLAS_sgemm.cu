#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

// Skeleton pulled from: https://siboehm.com/articles/22/CUDA-MMM

/*
 * A stand-alone script to invoke & benchmark standard cuBLAS SGEMM performance
 */
template <typename T>
void randomize_matrix(T *mat, int N) {
    // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
    // precision is too low and the same random number is generated.
    struct timeval time {};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        T tmp = (T)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

template <>
void randomize_matrix(__half *mat, int N) {
    // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
    // precision is too low and the same random number is generated.
    struct timeval time {};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = __float2half(tmp);
    }
}

template <typename T>
double diff(T a, T b) {
    double d = a - b;
    return d * d;
}

template <>
double diff(__half a, __half b) {
    double d = __half2float(a) - __half2float(b);
    return d * d;
}

template <typename T>
void gemm_test(int computeType, std::vector<int> &sizes) {
    bool verify = true;
    float elapsedTime;
    float totalTime;
    int print = 0;
    cublasStatus_t stat;    // cuBLAS functions status
    cublasHandle_t handle;  // cuBLAS context

    cudaDataType_t dtype;
    cublasComputeType_t ctype;
    cublasGemmAlgo_t alg = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    std::vector<std::string> computeTypeValues = {
        "NORMAL", "PEDANTIC", "FAST_TF32", "FAST_16F", "FAST_16BF"};

    std::map<int, std::vector<cublasComputeType_t>> computeTypes{
        {2, {CUBLAS_COMPUTE_16F, CUBLAS_COMPUTE_16F_PEDANTIC}},
        {4,
         {CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_PEDANTIC,
          CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_COMPUTE_32F_FAST_16F,
          CUBLAS_COMPUTE_32F_FAST_16BF}},
        {8, {CUBLAS_COMPUTE_64F, CUBLAS_COMPUTE_64F_PEDANTIC}}};

    std::map<int, cudaDataType_t> dataTypes{
        {2, CUDA_R_16F}, {4, CUDA_R_32F}, {8, CUDA_R_64F}};

    dtype = dataTypes[sizeof(T)];
    ctype = computeTypes[sizeof(T)][computeType];

    int nTrials = 10;

    stat = cublasCreate(&handle);  // initialize CUBLAS context

    for (const int &size : sizes) {
        double sum_mse = 0.0;
        totalTime = 0;
        int m = size;
        int k = size;
        int n = size;
        for (int trial = 0; trial <= nTrials; trial++) {
            cudaError_t cudaStat;  // cudaMalloc status

            T *a, *b, *c, *vc;
            // DEVICE
            T *d_a, *d_b, *d_c, *d_vc;

            // malloc for a,b,c...
            a = (T *)malloc(m * k * sizeof(T));
            b = (T *)malloc(k * n * sizeof(T));
            c = (T *)malloc(m * n * sizeof(T));
            vc = (T *)malloc(m * n * sizeof(T));

            randomize_matrix<T>(a, m * k);
            randomize_matrix<T>(b, k * n);
            randomize_matrix<T>(c, m * n);

            // cudaMalloc for d_a, d_b, d_c...
            cudaMalloc((void **)&d_a, m * k * sizeof(T));
            cudaMalloc((void **)&d_b, k * n * sizeof(T));
            cudaMalloc((void **)&d_c, m * n * sizeof(T));
            cudaMalloc((void **)&d_vc, m * n * sizeof(T));

            cudaEvent_t beg, end;
            cudaEventCreate(&beg);
            cudaEventCreate(&end);

            cudaMemcpy(d_a, a, m * k * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, b, k * n * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemset(d_c, 0, m * n * sizeof(T));
            cudaMemset(d_vc, 0, m * n * sizeof(T));

            T alpha = 1.0f;
            T beta = 0.5f;

            cudaEventRecord(beg);
            stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                &alpha, d_b, dtype, n, d_a, dtype, k, &beta,
                                d_c, dtype, n, ctype, alg);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                std::cout << "ERROR! " << std::endl;
            }

            cudaEventRecord(end);
            cudaEventSynchronize(beg);
            cudaEventSynchronize(end);

            elapsedTime = 0;
            cudaEventElapsedTime(&elapsedTime, beg, end);

            if (verify) {
                cudaMemcpy(c, d_c, m * n * sizeof(T), cudaMemcpyDeviceToHost);
                stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                    &alpha, d_b, dtype, n, d_a, dtype, k, &beta,
                                    d_vc, dtype, n, computeTypes[sizeof(T)][1],
                                    alg);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    std::cout << "ERROR! " << std::endl;
                }
                cudaMemcpy(vc, d_vc, m * n * sizeof(T), cudaMemcpyDeviceToHost);

                double sumdiff = 0.0f;
                for (int i = 0; i < m * n; i++) {
                    sumdiff = diff(c[i], vc[i]);
                }
                sum_mse += sumdiff / (m * n);
            }

            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
            cudaFree(d_vc);
            free(a);
            free(b);
            free(c);
            free(vc);

            if (trial > 0) {
                totalTime += elapsedTime;
            }
        }

        double flops = 2 * float(n) * float(m) * float(k);
        double gflopspers = (nTrials * (flops * 1e-9)) / (totalTime * 1e-3);
        std::map<int, std::string> prec{{2, "half"}, {4, "single"}, {8, "double"}};
        std::cout << prec[sizeof(T)] << "," 
                  << size << ","
                  << computeTypeValues[computeType] << ","
                  << sum_mse / nTrials << "," 
                  << gflopspers << std::endl;
    }

    cublasDestroy(handle);  // destroy CUBLAS context
}

int main(int argc, char *argv[]) {
    std::vector<int> sizesH = {32, 512, 1024, 2048, 4096};
    gemm_test<__half>(0, sizesH);
    gemm_test<__half>(1, sizesH);
    std::vector<int> sizesS = {32, 512, 1024, 2048, 4096};
    gemm_test<float>(0, sizesS);
    gemm_test<float>(1, sizesS);
    gemm_test<float>(2, sizesS);
    gemm_test<float>(3, sizesS);
    gemm_test<float>(4, sizesS);
    std::vector<int> sizesD = {2048};
    gemm_test<double>(0, sizesD);
    gemm_test<double>(1, sizesD);

    return EXIT_SUCCESS;
}
