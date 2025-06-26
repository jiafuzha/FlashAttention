/*
Kernels for attention forward pass.

Compile example:
nvcc -O3 --use_fast_math attention_forward.cu -o attention_forward -lcublas

version 1 is naive port from CPU code to kernel, parallelize over batch, time, heads only
./attention_forward 1

version 2 is a naive implementation of flash attention, taken, adapted from
https://github.com/tspeterkim/flash-attention-minimal
and with help from
https://github.com/leloykun/flash-hyperbolic-attention-minimal
sadly, this flash attention version seems about 3X slower than the naive version
./attention_forward 2

version 3 is a cuBLAS + softmax version, similar to the PyTorch implementation
cuBLAS is used both to calculate the QK^T and the final weighted sum
the softmax is calculated using a custom, efficient kernel as well
this turns out to be ~20X faster than (1) nice
./attention_forward 3

version 4 is a further optimized kernel that fuses the scale operation,
uses a directly autoregressive softmax, and uses the online softmax algorithm.
./attention_forward 4
*/

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"

#include <iostream>
#include <fstream>

// ----------------------------------------------------------------------------
// CUDA setup

// ----------------------------------------------------------------------------
// CPU code reference
double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}


void attention_forward_cpu(float* out, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }
                // pad with -INFINITY outside of autoregressive region for debugging comparisons
                for (int t2 = t+1; t2 < T; t2++) {
                    preatt_bth[t2] = -INFINITY;
                }

                // pass 2: calculate the exp and keep track of sum
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                // printf("\nq: %d, head: %d, attention score\n", t, h);
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                    // printf("  %f  ", att_bth[t2]);
                }


                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                // printf("\nq: %d, head: %d, output\n", t, h);
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
                // for (int i=0; i<hs; i++) {
                //     printf("  %f  ", out_bth[i]);
                // }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]

    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        // int rest = idx % (NH * N * d);
        // int nh_ = rest / (N * d);
        // rest = rest % (N * d);
        // int n = rest / d;
        // int d_ = rest % d;
        int nh_ = (idx / (N * d)) % NH;
        // assert(nh_ == nh_2);
        int n = (idx / (d)) % N;
        int d_ = idx % d;
        // assert (n == n_2);
        // assert (d_ == d_2);

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

__global__ void unpermute_kernel(const float* inp, float *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        // int rest = idx % (NH * N * d);
        int nh_ = (idx / (N * d)) % NH;
        // rest = rest % (N * d);
        int n = (idx / d) % N;
        int d_ = idx % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}


__global__ void flashattention2(float *out, float *K, float *Q, float* V, float scaling, int T_r, int T_c, int seq_len)
{   
     // used by attention_forward6
    // define constants, could be adjusted for different hardware specs
    const int d = 64;
    const int B_c = 32;
    const int B_r = 32;
    const int BK = B_c;
    // const int CACHE_Q = 1; // if 1 then cache Q in SMEM otherwise reload it over the tiles

    const int batch_offset = d * seq_len * blockIdx.x;
    const int TN = 8;
    const int TM = 4;

    // const int eles_per_output_row = (d+blockDim.x - 1)/blockDim.x;
    // const int eles_per_output_row = 8;
    // const int eles_per_output_row = 4;
    const int num_tiles = d/B_c; // or d/BK, number of tiles that the attention computation is split into
    /*
    NOTE: all are fully loaded into shared memory SMEM, I think we should adjust this as second step to only loading it in tiles of B_r x 32 
    and iterating the mults over the 32 sized tiles this way we can have a larger d, while keeping occupancy high
    */

    // statically define in SMEM and still address it with indices
    __shared__ float Q_i[B_r][d+1]; // uncomment only if you want to cache over full d (if CACHE_Q = 1)
    // __shared__ float Q_i[B_r][BK]; // if you want to save SMEM loads and keep the full Q loaded then change this to [B_r][d]
    
    __shared__ float K_j[B_c][BK+1]; // reduce SMEM bank conflicts by adding 1 column as K will be loaded transposed!
    __shared__ float V_j[B_c][BK];
    
    // attention result
    __shared__ float S_i[B_r][B_c+1]; // reduce SMEM bank conflicts by adding 1 column (in the naive softmax part)
    
    const uint totalResultsBlocktile = B_r * B_c; // number of results to calculate per block
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN); // number of threads needed
    const int threadId_flat = threadIdx.y * blockDim.x + threadIdx.x; // flattened thread id  (used for coalesced loading of tiles)

    // each thread process one block at position:
    // const int threadCol = threadId_flat / TN;
    // const int threadRow = threadId_flat / (B_r / TM);
    const int threadCol = threadId_flat % blockDim.x;
    const int threadRow = threadIdx.y;
    // const int threadRow = threadId_flat / blockDim.y;
        
    float l_i[TM]= {0.0};; // storing the intermediate sum of exponentials per row
    float m_i[TM]; // storing the intermediate max value of the rows
    float last_m[TM]; // storing the last max value of the rows
    // float O_i[num_tiles * TN * TM] = {0.0}; // storing the intermediate results of the Outputs (each thread stores a chunk TM x TN per tile)
    
    float O_i[num_tiles * TM * TN] = {0.0};
    
    // reset to min
    for (int ii = 0; ii < TM; ii++) {
        m_i[ii] = -INFINITY;
    }

    //WARNING: due to coalsecing I should probably add a second set of variables for using BK+1
    const uint strideK = numThreadsBlocktile / BK; // 64 / 64 = 1
    uint innerRowK = threadId_flat / BK; // 0-63 / 64, 0000000000000...0
    uint innerColK = threadId_flat % BK; // 0-63 % 64, 0123456789101112...63

    // int id;
    // load Q_i, UNCOMMENT only if your Q is caching over full d
    // const uint innerRowQ = threadId_flat / d; // 0-63 / 64, 0000000000000...0
    // const uint innerColQ = threadId_flat % d; // 0-63 % 64, 0123456789012...63
    // const uint nr_loads = B_r * d / numThreadsBlocktile;

    for (int t=0; t<(B_r * d); t+=numThreadsBlocktile){
      // need to load block of size B_r x d (64 x 64) with numThreadsBlocktile threads
      // if (blockIdx.y * B_r + innerRowQ) * d + innerColQ + t * numThreadsBlocktile / d
      int id = blockIdx.y * B_r * d + threadId_flat + t;
      uint innerRowQ = (threadId_flat + t) / d;
      uint innerColQ = (threadId_flat + t) % d;
      // 4 x 4 then this is 5 thus 5/
      if (id < d*seq_len){
        Q_i[innerRowQ][innerColQ] = Q[batch_offset + id];
      }
      else {
        Q_i[innerRowQ][innerColQ] = 0.0;
      }
    }

    __syncthreads();

    // scratchpad register for register-tiling (coarsening of the matrix mults)
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (int j = 0; j < T_c && j <= blockIdx.y ; j++) { // iterate of ver the chunks of K and V
        float threadResults[TM * TN] = {0.0}; // storing the intermediate outputs
        
        for (int t=0; t<num_tiles; t++){
            for (int i=0; i<B_c; i+=strideK) {
                int id = (innerRowK + j * B_c) * d + i * d + innerColK + t * BK;
                if (id < d*seq_len){
                    K_j[innerRowK+i][innerColK] = K[batch_offset + id];
                } else {
                    K_j[innerRowK+i][innerColK] = 0.0;
                }
            }
            __syncthreads();
        
            // for (int dd=0; dd<BK; dd++){ // load elements of Q_i and K_j^T into registers
            for (int dd=0; dd<BK; dd++){ // load elements of Q_i and K_j^T into registers
                for (uint i = 0; i < TM; ++i) {
                    regM[i] = Q_i[(threadRow * TM + i)][t*BK + dd]; // uncomment if you cache Q over full d
                }
                for (uint i = 0; i < TN; ++i) {
                    regN[i] = K_j[threadCol * TN + i][dd];
                }
                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                    }
                }
            }
            __syncthreads();
        }
        

        // store the results in S_i, account for causal masking
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                if (j*B_c + threadCol * TN + resIdxN <= blockIdx.y * B_r + threadRow * TM + resIdxM){
                    S_i[(threadRow * TM + resIdxM)][threadCol * TN + resIdxN] = threadResults[resIdxM * TN + resIdxN] *scaling;
                } else {
                    S_i[(threadRow * TM + resIdxM)][threadCol * TN + resIdxN] = -INFINITY;
                }      
            }
        }
        __syncthreads();

        for (int i=0;i<TM;++i){
            last_m[i] = m_i[i];
            float m = m_i[i];
            for (int jj = 0; jj < B_c; jj += 1) {
                if (m < S_i[threadRow*TM+i][jj]) {
                    m = S_i[threadRow*TM+i][jj];
                }
            }
            m_i[i] = m;
        }

        // 2) renormalize current O
        if (j > 0) {
            for (int t = 0; t < num_tiles; t++){
                for (int i=0;i<TM;++i){
                    // for (int jj=0;jj<TN;++jj){
                        // O_i[t*TN*TM + i*TN + jj] *= exp(last_m[i] - m_i[i]);
                    for (int jj=0; jj<TN; jj++){
                        // if (threadCol * TN + jj < BK) {
                            O_i[t*TM*TN + i*TN + jj] *= exp(last_m[i] - m_i[i]);
                        // }
                    }
                }
            }
        }

        // 3) renormalize the sum l_i
        for (int i=0;i<TM;++i){
            l_i[i] *= exp(last_m[i] - m_i[i]);
        }      

        for (int t = 0; t < num_tiles; t++){
            // load V
            __syncthreads();

            for (int i=0; i<B_c; i+=strideK){
                int id = (innerRowK + j * B_c) * d + i * d + innerColK + t * BK;
                if (id < d*seq_len){
                    V_j[innerRowK+i][innerColK] = V[batch_offset + id];
                } else {
                    V_j[innerRowK+i][innerColK] = 0.0;
                }
            }
             __syncthreads();

            for (int dd = 0; dd < B_c; dd++) {
                for (int ii = 0; ii < TM; ii++){
                    regM[ii] = exp(S_i[threadRow*TM+ii][dd] - m_i[ii]);
                    if (t==0){
                        // add all elements per row
                        l_i[ii] += regM[ii];
                    }
                    // regN[ii] = V_j[dd][threadCol * TN + ii];
                }
                for (int ii=0;ii<TM;ii++){
                    for (int jj=0; jj<TN; jj++){ // calculate output elements
                        if (threadCol * TN + jj < BK) {
                            // regN[jj] = V_j[dd][jj];
                            // O_i[t*TN*TM + ii*TN + jj] += regM[ii] * regN[jj];
                            O_i[t*TM*TN + ii*TN + jj] += regM[ii] * V_j[dd][threadCol * TN + jj];
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // normalize by the output sum and write to out matrix
    for (int t = 0; t < num_tiles; t++){
        for (int ii=0;ii<TM;ii++){
            for (int jj=0; jj<TN; jj++){
                if(blockIdx.y*B_r+threadRow*TM+ii < seq_len){
                    // out[batch_offset + (blockIdx.y * B_r + threadRow*TM + ii) * d + t * BK + threadCol*TN + jj] = O_i[t*TN*TM+ii*TM+jj] / l_i[ii];
                    out[batch_offset + (blockIdx.y * B_r + threadRow*TM + ii) * d + t*BK + threadCol*TN + jj] = O_i[t*TM*TN + ii*TN + jj] / l_i[ii];
                }
            }
        } 
    }
}

template <int HEAD_SIZE, int BLOCK_ROWS, int BLOCK_COLS>
__global__ void flashattention_warps(float *out, float *K, float *Q, float* V, float scaling, int seq_len)
{   
    const int batch_offset = HEAD_SIZE * seq_len * blockIdx.x;

    __shared__ float Q_i[BLOCK_ROWS][HEAD_SIZE]; // uncomment only if you want to cache over full d (if CACHE_Q = 1)
    // __shared__ float Q_i[B_r][BK]; // if you want to save SMEM loads and keep the full Q loaded then change this to [B_r][d]
    
    __shared__ float K_j[BLOCK_COLS][HEAD_SIZE+1]; // reduce SMEM bank conflicts by adding 1 column as K will be loaded transposed!
    __shared__ float V_j[BLOCK_COLS][HEAD_SIZE];

    __shared__ float O_i[BLOCK_ROWS][HEAD_SIZE+1]; 
    
    // attention result
    // __shared__ float S_i[BLOCK_ROWS][BLOCK_COLS]; // reduce SMEM bank conflicts by adding 1 column (in the naive softmax part)

    const int thread_flatid = threadIdx.y * blockDim.x + threadIdx.x;

    const int lane = thread_flatid % warpSize;
    // const int warp_idx = thread_flatid / warpSize;

    // const int num_warps = NUM_THREADS / warpSize;
    const int NUM_THREADS = blockDim.x * blockDim.y * blockDim.z;

    // thread block TM * TN for caching and reusing data inside register
    const int TM = 16/sizeof(float);
    const int TN = TM;

    // const int TM=2;
    // const int TN=2;

    const int MAX_SHUFFLE_MASK = BLOCK_COLS/TN;

    // for warp level shuffle
    // const int NUM_THREAD_BLOCKS_PER_BLOCK_COLS = BLOCK_COLS/TN;
    // const int NUM_THREAD_BLOCKS_PER_BLOCK_ROWS = BLOCK_ROWS/TM;

    float regTM[TM];
    float regTN[TN];
    float m_i[TM];
    float l_i[TM] = {0.0};
    // float to_i[TM] = {0.0};
    float last_li[TM];

    // printf("I am here \n");

    // constexpr int HEAD_VEC_SIZE_PER_THREAD = 16/sizeof(float);

    // using VEC_TYPE = typename Vec<float, HEAD_VEC_SIZE_PER_THREAD>::Type;

    // const int NUM_ITERS_PER_HEAD_DIM = HEAD_SIZE / HEAD_VEC_SIZE_PER_THREAD; // 16 bytes at a time for each thread
        
    // float l_i[BLOCK_ROWS]= {0.0};; // storing the intermediate sum of exponentials per row
    // float m_i[BLOCK_ROWS]; // storing the intermediate max value of the rows
    // float last_m[BLOCK_ROWS]; // storing the last max value of the rows
    // float O_i[num_tiles * TN * TM] = {0.0}; // storing the intermediate results of the Outputs (each thread stores a chunk TM x TN per tile)
    
    // float O_i[num_tiles * TM * TN] = {0.0};

    
    // reset to min
    for (int ii = 0; ii < TM; ii++) {
        m_i[ii] = -INFINITY;
    }

    int offset = batch_offset + blockIdx.y * BLOCK_ROWS * HEAD_SIZE;

    // printf("Q_i: \n");

    for (int i=thread_flatid; i<BLOCK_ROWS*HEAD_SIZE; i+=NUM_THREADS) {
        int id = offset + i;
        int row = i / HEAD_SIZE;
        int col = i % HEAD_SIZE;
        Q_i[row][col] = Q[id];
        // if (col == 0) {
        //     printf("\n");
        // }
        // printf("block: <%d %d>, thread: <%d, %d>, flatid: %d, [%d, %d]= %f, ", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, thread_flatid, row, col, Q_i[row][col]);
        
        O_i[row][col] = 0.0;
    }


    int T_c = (seq_len + BLOCK_COLS - 1) / BLOCK_COLS;

    // __syncthreads();
    for (int jj = 0; jj < T_c && jj <= blockIdx.y; jj++) {
        offset = batch_offset + jj * BLOCK_COLS * HEAD_SIZE;
        for (int i=thread_flatid; i<BLOCK_COLS*HEAD_SIZE; i+=NUM_THREADS) {
            int row = i / HEAD_SIZE;
            int col = i % HEAD_SIZE;
            K_j[row][col] = K[offset + i];
            V_j[row][col] = V[offset + i];
        }
        __syncthreads();
        // qk in warp
        float threadResults[TM * TN] = {0.0}; // storing the intermediate outputs
        float last_mi[TM];
        // float row_sum[TM] = {0.0};
        for (int dd=0; dd<HEAD_SIZE; dd++) {
            for (int i=0; i<TM; i++) {
                regTM[i] = Q_i[threadIdx.y*TM + i][dd];
            }
            
            for (int i=0; i<TN; i++) {
                regTN[i] = K_j[threadIdx.x*TN + i][dd];
            }

            for (int i=0; i<TM; i++) {
                for (int j=0; j<TN; j++) {
                    threadResults[i*TN +j] += regTM[i] * regTN[j];
                }
            }
        }
        for (int i=0; i<TM; i++) {
            last_mi[i] = m_i[i];
            for (int j=0; j<TN; j++) {
                if (jj*BLOCK_COLS + threadIdx.x * TN + j <= blockIdx.y * BLOCK_ROWS + threadIdx.y * TM + i) {
                    threadResults[i*TN + j] *= scaling;
                } else {
                    threadResults[i*TN + j] = -INFINITY;
                }
                if (m_i[i] < threadResults[i*TN + j]) {
                    m_i[i] = threadResults[i*TN + j];
                }
            }
        }
        // warp level shuffle to get max in blocks
        for (int i=0; i<TM; i++) {
            for (int m=MAX_SHUFFLE_MASK/2; m>=1; m/=2) {
                m_i[i] = fmaxf(m_i[i], __shfl_xor_sync((uint32_t)-1, m_i[i], m));
            }
        }
        // broadcast mi
        for (int i=0; i<TM; i++) {
            m_i[i] = __shfl_sync((uint32_t)-1, m_i[i], 0, MAX_SHUFFLE_MASK);
        }
        // ==============
        // softmax
        for (int i=0; i<TM; i++) {
            if ((lane % MAX_SHUFFLE_MASK) == 0) {
                last_li[i] = l_i[i];
            }
            l_i[i] = 0.0;
            for (int j=0; j<TN; j++) {
                // p
                threadResults[i*TN + j] = exp(threadResults[i*TN + j] - m_i[i]);
                l_i[i] += threadResults[i*TN + j];
            }
        }
        // shuffle row sum
        for (int i=0; i<TM; i++) {
            for (int m=MAX_SHUFFLE_MASK/2; m>=1; m/=2) {
                l_i[i] += __shfl_xor_sync((uint32_t)-1, l_i[i], m);
            }
        }
        // broadcast row sum
        for (int i=0; i<TM; i++) {
            if (jj>0 && (lane % MAX_SHUFFLE_MASK) == 0) {
                l_i[i] += last_li[i] * exp(last_mi[i] - m_i[i]);
            }
            // l_i[i] = __shfl_sync((uint32_t)-1, l_i[i], 0, MAX_SHUFFLE_MASK);
        }
        
        // PV
        for (int dd=0; dd<HEAD_SIZE; dd++) {
            float to_i[TM] = {0.0};
            // along BLOCK_COLS direction
            for (int j=0; j<TN; j++) {
                if (threadIdx.x*TN + j < BLOCK_COLS) {
                    regTN[j] = V_j[threadIdx.x*TN + j][dd];
                }
            }
            for (int i=0; i<TM; i++) {
                for (int j=0; j<TN; j++) {
                    if (threadIdx.x*TN + j < BLOCK_COLS) {
                        to_i[i] += threadResults[i*TN +j] * regTN[j];
                    }
                }
            }
            // shuffle to get sum
            // all threads in warp complete all BLOCK_COLS elements in one iteration per head dim
            for (int i=0; i<TM; i++) {
                for (int m=MAX_SHUFFLE_MASK/2; m>=1; m/=2) {
                    to_i[i] += __shfl_xor_sync((uint32_t)-1, to_i[i], m);
                }
            }
            if ((lane % MAX_SHUFFLE_MASK) == 0) {
                for (int i=0; i<TM; i++) {
                    if (jj > 0) {
                        // no need O_i?
                        // O_i[threadIdx.y*TM + i][dd] = O_i[threadIdx.y*TM + i][dd] * exp(m_i[i] - last_mi[i]) + to_i[i];
                        O_i[threadIdx.y*TM + i][dd] = O_i[threadIdx.y*TM + i][dd] * exp(last_mi[i] - m_i[i]) + to_i[i];
                    } else {
                        O_i[threadIdx.y*TM + i][dd] = to_i[i];
                    }
                }
            }
        }
    }
    if ((lane % MAX_SHUFFLE_MASK) == 0) {
        for (int dd=0; dd<HEAD_SIZE; dd++) {
            for (int i=0; i<TM; i++) {
                // for (int j=0; j<TN; j++) {
                    
                if(blockIdx.y*BLOCK_ROWS+threadIdx.y*TM+i < seq_len){ 
                    out[batch_offset + (blockIdx.y * BLOCK_ROWS + threadIdx.y * TM + i) * HEAD_SIZE + dd] = O_i[threadIdx.y*TM + i][dd] / l_i[i];
                }
                // }
            }
        }
    }


    // normalize by the output sum and write to out matrix
    // for (int t = 0; t < num_tiles; t++){
    //     for (int ii=0;ii<TM;ii++){
    //         for (int jj=0; jj<TN; jj++){
    //             if(blockIdx.y*B_r+threadRow*TM+ii < seq_len){
    //                 // out[batch_offset + (blockIdx.y * B_r + threadRow*TM + ii) * d + t * BK + threadCol*TN + jj] = O_i[t*TN*TM+ii*TM+jj] / l_i[ii];
    //                 out[batch_offset + (blockIdx.y * B_r + threadRow*TM + ii) * d + t*BK + threadCol*TN + jj] = O_i[t*TM*TN + ii*TN + jj] / l_i[ii];
    //             }
    //         }
    //     } 
    // }
}


void attention_forward7(float* out,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
   
}

// template <int, int, int>
// __global__ void flashattention_warps(float *out, float *K, float *Q, float* V, float scaling, int seq_len);
template <int B_r, int B_c, int d>
void attention_forward8(float* out,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // these are hardcoded to 32 for now
    // constexpr int B_r = 4;
    // constexpr int B_c = 4;
    // renaming these to be consistent with the kernel
    // const int B = B;
    const int nh = NH;
    const int N = T;
    assert(d == C / NH);

    // const int NUM_THREADS = 128;
    // const int BLOCK_ROWS = 32;

    // const int NUM_THREADS_PER_ROW = NUM_THREADS / BLOCK_ROWS;

    const float softmax_scale = 1.0 / sqrt(d);

    // calculate SRAM size needed per block, ensure we have enough shared memory
    // int col_tile_size = B_r * d;  // size of Kj, Vj
    // int row_tile_size = B_c * d;  // size of Qi
    // const int sram_size =
    //     (col_tile_size * sizeof(float))  // SRAM size for Vj
    //     + (row_tile_size * sizeof(float))  // SRAM size for Qi
    //     + (B_r * (B_c+1) * sizeof(float)) // SRAM size for S
    //     + (col_tile_size * sizeof(float)); // SRAM size for Kj, 

    // int max_sram_size;
    // cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    // const int B_r = max_sram_size / (4*d)
    // const int B_c = B_r;


    // if (sram_size > max_sram_size) {
    //     printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);
    //     printf("SRAM size exceeds maximum shared memory per block\n");
    //     printf("Try decreasing col_tile_size or row_tile_size further\n");
    //     exit(1);
    // }

    // okay so now, this kernel wants Q,K,V to all be of shape (B, nh, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, nh, d)
    // so we have to permute the tensor using a kernel with block_size
    float *q, *k, *v;
    cudaCheck(cudaMalloc(&q, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&k, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&v, B * T * C * sizeof(float)));

    const int TM = 16/sizeof(float);
    const int TN = TM;

    // const int TM=2;
    // const int TN=2;

    dim3 blockDim(B_c/TN, B_r/TM);
    dim3 gridDim(B*nh, (N+B_r-1)/B_r);

    int total_threads = B * N * nh * d;
    int num_blocks = ceil_div(total_threads, block_size);
    
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, N, nh, d);

    // float * cpu_q = (float*)malloc(B * T * C * sizeof(float));

    // cudaCheck(cudaMemcpy(cpu_q, q, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));

    // int offset;

    // // copy back q and print
    // for (int i=0; i<nh; i++) {
    //     // offset = i * N * d;
    //     for (int j=0; j<N/B_r; j++) {
    //         // offset += j * B_r * d;
    //         for (int k=0; k<B_r; k++) {
    //             // offset += k * d;
    //             printf("head idx: %d, group idx: %d, B_r idx: %d, ", i, j, k);
    //             for (int z=0; z<d; z++) {
    //                 offset = i*N*d + j * B_r *d + k * d + z;
    //                 printf("[%d, %d]=%f, ", k, z, cpu_q[offset]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    // free(cpu_q);

    // now actually call the flash attention kernel
    cudaDeviceSynchronize();
    double start, end;
    start = getTimeStamp();
    flashattention_warps<d, B_r, B_c>
            <<<gridDim, blockDim>>>(out, k, q, v, softmax_scale, N);
    cudaError_t code = cudaPeekAtLastError();
    printf("kernel launch result: %d\n", code);
    if (code != cudaSuccess) {
        printf("error: %s\n", cudaGetErrorString(code));
    }
    cudaDeviceSynchronize();
    end = getTimeStamp();
    printf("Time taken for attention kernel: %f\n", end-start);

    // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    num_blocks = ceil_div(B * T * C, block_size);

    unpermute_kernel<<<num_blocks, block_size>>>(out, q, B, N, nh, d);
    cudaDeviceSynchronize();
    cudaCheck(cudaMemcpy(out, q, B * T * C * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaDeviceSynchronize();
    // free memory
    cudaCheck(cudaFree(q));
    cudaCheck(cudaFree(k));
    cudaCheck(cudaFree(v));
}

// kernel version dispatch
template <int B_r, int B_c, int d>
void attention_forward(int kernel_num,
                       float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    switch (kernel_num) {
        case 7:
            attention_forward7(out, inp, B, T, C, NH, block_size);
            break;
        case 8:
            attention_forward8<B_r, B_c, d>(out, inp, B, T, C, NH, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}
// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 6;
    int T = 4096;
    constexpr int C = 1536;
    constexpr int NH = 12;

    // constexpr int C = 16;
    // constexpr int NH = 2;

    constexpr int br=16;
    constexpr int bc=16;
    constexpr int d = C/NH;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    // cublasCreate(&cublas_handle);

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    float* inp = make_random_float(B * T * 3 * C);

    // move to GPU
    float* d_out;
    float* d_vaccum;
    float* d_qkvr;
    float* d_preatt;
    float* d_att;
    float* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_vaccum, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_qkvr, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_preatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * 3 * C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);
    // int block_sizes[] = {32, 64, 128, 256, 512};
    int block_sizes[] = {128};

    // first check the correctness of the kernel

    // try to read out from disk frist
    std::string filename("cpu.out.bin");
    std::ifstream cpuOutFile(filename, std::ios::binary);
    if (cpuOutFile.is_open()) {
        cpuOutFile.read(reinterpret_cast<char*>(out), B * T * C * sizeof(float));
        cpuOutFile.close();
    } else {
        attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);
        std::ofstream newCpuOutFile(filename, std::ios::binary);
        newCpuOutFile.write(reinterpret_cast<const char*>(out), B * T * C * sizeof(float));
        newCpuOutFile.close();
    }

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        attention_forward<br, bc, d>(kernel_num, d_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T, C, NH, block_size);
        // all kernels should produce the correct output out
        validate_result(d_out, out, "out", B * T * C, 1e-4f);
        // but as for preatt and att, things get a bit more complicated:
        // if (kernel_num != 2 && kernel_num != 6) {
            // kernel 2 (knowingly) fails att/preatt because it uses a different algorithm
            // that estimates the softmax online and never materializes preatt/att
        //     validate_result(d_att, att, "att", B * NH * T * T, 1e-4f);
        // }
        // if (kernel_num != 2 && kernel_num != 4 && kernel_num != 5 && kernel_num != 6) {
            // kernel 4 (knowingly) fails preatt because it fuses the scale normalization
            // into the softmax, so preatt is off by 1.0f / sqrt(HS)
            // but att and out (checked below) should match.
            // validate_result(d_preatt, preatt, "preatt", B * NH * T * T, 1e-4f);
        // }
    }
    printf("All results match. Starting benchmarks.\n\n");

    // benchmark speed of the kernel
    // for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    //     int block_size = block_sizes[j];
    //     int repeat_times = 100;

    //     float elapsed_time = benchmark_kernel(repeat_times, attention_forward,
    //                                           kernel_num, d_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp,
    //                                           B, T, C, NH, block_size);

    //     printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
    // }

    // free memory
    free(out);
    free(preatt);
    free(att);
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_vaccum));
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_preatt));
    cudaCheck(cudaFree(d_att));
    cudaCheck(cudaFree(d_inp));
    // cublasDestroy(cublas_handle);

    return 0;
}