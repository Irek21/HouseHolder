#include <mpi.h>
#include <fstream>
#include "Lib/declares.h"
#include "Lib/householder.h"

int seed(int commRank, int commSize, int timeSeed)
{
	int *seeds = NULL;
	if (commRank == 0) {
		srand(timeSeed);
		seeds = new int[commSize];
		if (seeds == NULL) {
			return -1;
		}
		for (int i = 0; i < commSize; i++) {
			seeds[i] = rand();
		}
	}
	int seed;
	MPI_Scatter(seeds, 1, MPI_INTEGER, &seed, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
	if (commRank == 0) {
		delete[] seeds;
	}
	srand(seed);
	return 0;
}

Vector &gatherX(CycleMatrix &A, Vector &x)
{
	int N = A.size, commRank = A.commRank, commSize = A.commSize, localM = A.localM;
	int residual = N % commSize;
	Vector gatherX(N);

	int recvCounts[commSize];
	for (int i = 0; i < commSize; ++i) recvCounts[i] = (N / commSize);
	for (int i = 0; i < residual; ++i) recvCounts[i] += 1;

	int displs[commSize];
	displs[0] = 0;
	for (int i = 1; i < commSize; ++i) displs[i] = recvCounts[i - 1] + displs[i - 1];

	int xSize = 0;
	if (commRank == residual) { xSize = x.size - 1; }
	else xSize = x.size;
	MPI_Gatherv(x.data, xSize, MPI_DOUBLE, gatherX.data, recvCounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (commRank == 0) {
		for (int i = 0; i < (N / commSize); ++i) {
			for (int j = 0; j < commSize; ++j) {
				cout << setw(13) << gatherX.data[i + displs[j]]; // << endl;
			}
		}

		for (int i = 0; i < residual; ++i) {
			cout << setw(13) << gatherX.data[N / commSize + displs[i]]; //  << endl;
		}
	}
	if (commRank == 0) cout << endl;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int commRank, commSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
	int timeSeed = time(NULL);
    seed(commRank, commSize, timeSeed);
    int N = atoi(argv[1]);
	double start = 0, T1 = 0, T2 = 0;

    CycleMatrix A(N, commRank, commSize);
	A.gen();
	if (N < 19) {
		if (commRank == 0) cout << "A:" << endl;
		A.print();
		if (commRank == 0) cout << endl;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (commRank == 0) start = MPI_Wtime();
	householder(A);
	MPI_Barrier(MPI_COMM_WORLD);
	if (commRank == 0) {
		T1 = MPI_Wtime() - start;
		cout << "T1 = " << T1 << endl << endl;
	}
	if (N < 19) {
		if (commRank == 0) cout << "A factorized:" << endl;
		A.print();
		if (commRank == 0) cout << endl;
	}

	Vector x(A.localM);
	MPI_Barrier(MPI_COMM_WORLD);
	if (commRank == 0) start = MPI_Wtime();
	solve(A, x);
	MPI_Barrier(MPI_COMM_WORLD);
	if (commRank == 0) {
		T2 = MPI_Wtime() - start;
		cout << "T2 = " << T2 << endl << endl;
	}
	if (N < 19) {
		if (commRank == 0) cout << "Decision:" << endl;
		gatherX(A, x);
	}

	seed(commRank, commSize, timeSeed);
	A.gen();
	double resNorm = residual(A, x);
	if (commRank == 0) cout << "Residual = " << resNorm << endl;
	if (commRank == 0) {
		ofstream fout("Out/res.txt", ios_base::app);
		fout << commSize << " " << T1 << " " << T2 << " " << T1 + T2 << endl;
		fout.close();
	}
    MPI_Finalize();
}
