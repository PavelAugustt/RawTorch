#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
# define N 1000 
#include <iomanip>

__global__ void d_transpose(int *d_a, int *d_c, int rows, int columns)
{
    // here even if we flip it, shit doesnt matter cus at the end of the day all we need is
    // every single combination of row id and column id for atleast one thread to do thecorresponding calculation for an accurate transpose.
    int columnId = threadIdx.x + blockDim.x * blockIdx.x;
    int rowId = threadIdx.y + blockDim.y * blockIdx.y;

    while(rowId < rows && columnId < columns)
    {
        d_c[rowId * columns + columnId] = d_a[columnId * rows + rowId];

        columnId += gridDim.x * blockDim.x;
        rowId += gridDim.y * blockDim.y;
    }
}

void print_matrix(const int *matrix, int rows, int cols, const std::string& name) {
    std::cout << name << ":\n";
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            // i think this is pretty simple too tbh, i could write this myself next time for repeittion tmrw
            // all it is doing is printing each element in row major order sequentially with spaces and line breaks
            std::cout << std::setw(4) << matrix[row * cols + col] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void transpose(int *A, int *B, int rows, int columns)
{
    for(int rowId = 0; rowId < rows; rowId++ )
    {
        for(int columnId = 0; columnId < columns; columnId++)
        {
            // here we're basically saying that each column-major id for our matrix would be the 
            // main matrix's row major id, which in other words flips i, j to j, i correspondingly
            // the formula in itself is pretty smart when it comes to flattening a multidimensional matrix into 1d 
            // baically in row major order or rowId * column + columnId
            // we basically say jump to the row our element is at with rowId * columns, cus columns = number of elements in a row
            // when u multiply that with the column id u jump across that many columns to our column, once u jump there when u add
            // col id to the mix u basically jump to the id of our element within that row, thats basically it.
            B[columnId * rows + rowId] = A[rowId * columns + columnId]; 
        }
    }
}
int main(void)
{
    size_t size = N * N * sizeof(int);
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);

    int *h_c = (int*)malloc(size);

    int *d_a, *d_c;

    for(int i = 0; i < N * N; i++)
    {
        h_a[i] = rand() % 100;
    }
    print_matrix(h_a, N, N, "Original Matrix A");
    transpose(h_a, h_b, N, N);

    print_matrix(h_b, N, N, "Transposed Matrix B");

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // initially my blockDim was kinda problematic with N, N, 0
    // Tomorrow we have to figure out how to break this thing, what if row, col weretn equal waht if it were greater what if it was lower eh and eh
    // figure out how stride loop would work for the kernel and so on, now im tired.
    // stick to hard core 256 and ceil for grid dim bingo

    // i think the gridDim and blockDim stuff is the most idiotic and problematic 

    // basically when u say launch a block with 16, 16 threads ur launching 256 threads in other words i want u to look at it like  amatrix
    // instead of looking at it 16 threads in x and y, in a 16 x 16 matrix although there are 16 threads in each direction
    // there are elements in that area which is 256 thats howthe threads are structured if u get what i mean 
    dim3 blockDim(16, 16, 1);

    // finally this is really important initially i was launching gridDim 1 dimensionally by ceil-ing for 256 threads directly across 1 dimension
    // why wont this work? essentially it wont cus look here threadIdx.y + blockDim.y * blockIdx.y;, when u do this itll launch blocks
    // across only x axis so blockDim across y axis will always be 0, so essentially when each block processes our matrix in 16 x 16 tiles
    // if our matrix goes beyond the 16 x 16 chunks it wont basically go beyond the first row/column depending on across what axis we're considering its still a bit hazy to me
    // might blackbox this
    dim3 gridDim(N+15/16, N+15/16, 1);  
    d_transpose<<<gridDim, blockDim>>>(d_a, d_c, N, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    print_matrix(h_c, N, N, "Transposed Matrix C");

    cudaFree(d_a);
    cudaFree(d_c);
    
    free(h_a);
    free(h_b);

    return 0;
}