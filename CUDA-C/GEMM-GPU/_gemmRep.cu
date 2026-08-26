# include <iostream>

# define N 2

void gemmCPU(int *a, int *b,  int *c, int columns, int rows)
{
	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < columns; j++)
		{
			for(int k = 0; k < columns; k++)
			{
				// we basically get c[i, j]
				// next look at matrix multiplication this was m x k, k x n
				// in thos eshapes the output matrix is always determined by m x n even though k==k

				// in other words for each row in matrix 1 we multiply it with each column in matrix 2
				// this is pretty intuitive but in code what we do is we keep row as 0, and we iterate thru each column in our second matrix and dot product them
				// ofc this would only apply for the number of eleemnts in matrix 1s row, in other words k < total number of columns of matrix which is k
				// which in other words would be the total number of rows in matrix 2 in that case we just jump to each row for the dot product would be something like
				// 0, 0, then 1, 0 which is the right wa for the second matrix, next we offset the column by j

				//will look into this for nonsquare matrices
				c[i * columns + j] = a[i * columns + k] * b[k * rows + j];
			}
		}
	}
}

void printMatrix(int *a, int columns, int rows)
{
	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < columns;  j++)
		{
			printf(" %d ", a[i * columns + j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(void)
{
	int *a, *b, *c;

	size_t size = N * N * sizeof(int);

	a = (int*) malloc(size);
	b = (int*) malloc(size);
	c = (int*) malloc(size);

	for(int i = 0; i < N * N; i++)
	{
		a[i] = i;
		b[i] = i + 1;
	}

	printMatrix(a, N, N);
	printMatrix(b, N, N);

	gemmCPU(a, b, c, N, N);
	printMatrix(c, N, N);

	
	free(a);
	free(b);
	free(c);

	return 0;
}
