#include<stdio.h>
#include<bits/stdc++.h>
#include<cuda_runtime.h>
#include<time.h>

using namespace std;

#define rep(i,a,b) for(int i = a;i<b;i++)
#define BLOCK_WIDTH 32

__global__ void matMulTiled(int* A, int* B, int* C,int m,int n, int p)
{
    __shared__ int tileA[BLOCK_WIDTH][BLOCK_WIDTH], tileB[BLOCK_WIDTH][BLOCK_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x, ty = threadIdx.y;
    int temp_sum = 0;
    tileA[ty][tx] = 0, tileB[ty][tx] = 0;

    rep(i,0,(p+BLOCK_WIDTH-1)/BLOCK_WIDTH)
    {
        if(row < m &&  (tx + i*BLOCK_WIDTH) < p)
            tileA[ty][tx] = A[row*p + i*BLOCK_WIDTH + tx];
        else
            tileA[ty][tx] = 0;

        if(col < n && (ty + i*BLOCK_WIDTH) < p)
            tileB[ty][tx] = B[col + (ty + i*BLOCK_WIDTH)*n ];
        else
            tileB[ty][tx] = 0;

        __syncthreads();

        rep(i,0,BLOCK_WIDTH)
            temp_sum += tileA[ty][i]*tileB[i][tx];
    }

    if((row<m)&&(col<n))
    {
        C[row*n + col] = temp_sum;
    }
}

void init(int* A,int* B,int m,int n,int p)
{
    rep(i,0,m)
    {   
        rep(j,0,p)
            *(A+i*p+j) = 1;
    }

    rep(i,0,p)
    {   
        rep(j,0,n)
             *(B+i*n+j) = 2;
    }
   
}

int main()
{
    int *h_a, *h_b, *h_c; //host pointers
    int *d_a, *d_b, *d_c; //device pointers
    
    int m,p,n;

    m = 4, p = 2, n = 4; 

    size_t bytes_a = m*p*sizeof(int);
    size_t bytes_b = n*p*sizeof(int);
    size_t bytes_c = m*n*sizeof(int);

    h_a = (int*)malloc(bytes_a);
    h_b = (int*)malloc(bytes_b);
    h_c = (int*)malloc(bytes_c);

    init(h_a,h_b,m,n,p);

    cudaMalloc(&d_a,bytes_a);
    cudaMalloc(&d_b,bytes_b);
    cudaMalloc(&d_c,bytes_c);

    int block_size = 32;
    int grid_size = (int)ceil((float)64/block_size);

    

    dim3 grid(grid_size,grid_size);
    dim3 threads(block_size,block_size);

    // cout<<block_size<<" "<<grid_size<<"\n";

    cudaMemcpy(d_a,h_a,bytes_a,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,bytes_b,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,h_c,bytes_c,cudaMemcpyHostToDevice);

    

    matMulTiled<<<grid,threads>>> (d_a,d_b,d_c,m,n,p);

    cudaMemcpy(h_c,d_c,bytes_c,cudaMemcpyDeviceToHost);

    rep(i,0,m)
    {
        rep(j,0,n)
            cout<<*(h_c+i*n+j)<<" ";
        cout<<"\n";
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}