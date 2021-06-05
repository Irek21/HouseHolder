#include <cmath>
#include "declares.h"

void countPivot(Vector &a, Vector &x, int K) {
    int N = a.size;
    double Sk = 0;
    for (int i = K + 1; i < N; ++i) {
        Sk += a.data[i] * a.data[i];
        x.data[i] = a.data[i];
    }

    double aNorm = sqrt(Sk + a.data[K] * a.data[K]);
    x.data[K] = a.data[K] - aNorm;
    double xNorm = sqrt(Sk + x.data[K] * x.data[K]);
    for (int i = K; i < N; ++i) {
        x.data[i] /= xNorm;
        a.data[i] = 0;
    }
    a.data[K] = aNorm;
}

void matMul(CycleMatrix &A, CycleMatrix &tmpA, Vector &U, Vector &x, int K)
{
    int N = U.size;
    int commRank = A.commRank, commSize = A.commSize;
    int rowK = (commRank > (K % commSize)) ? (K / commSize) : (K / commSize + 1); // CHECK THIS MOMENT !

    for (int i = K; i < N; ++i) {
        for (int t = K; t < N; ++t) U.data[t] = -2 * x.data[i] * x.data[t];
        U.data[i] += 1;
        for (int k = rowK; k < A.localM; ++k) {
            for (int j = K; j < N; ++j) {
                tmpA.data[k][j] += U.data[j] * A.data[k][i];
            }
        }
    }

    for (int i = rowK; i < A.localM; ++i) {
        for (int j = K; j < N; ++j) {
            A.data[i][j] = tmpA.data[i][j];
            tmpA.data[i][j] = 0;
        }
    }
}

void householder(CycleMatrix &A)
{
    int N = A.size, commRank = A.commRank, commSize = A.commSize;
    int residual = N % commSize,
        stripes = residual == 0 ? N / commSize : N / commSize + 1; // N + 1 !

    Vector a(N);
    Vector x(N);
    Vector U(N);
    CycleMatrix tmpA(N, commRank, commSize);

    int K = 0;
    for (int i = 0; i < stripes - 1; ++i) {
        for (int j = 0; j < commSize; ++j) {
            // if (i * commSize + j == N - 1) break;
            if (commRank == j) {
                A.readColumn(a, i);
                countPivot(a, x, K);
            }
            MPI_Bcast(&x.data[K], N - K, MPI_DOUBLE, j, MPI_COMM_WORLD);

            matMul(A, tmpA, U, x, K);
            if (commRank == j) A.writeColumn(a, i); // CHECK THIS MOMENTS !

            K++;
        }
    }

    // residual = residual == 0 ? commSize : residual;
    if (residual > 0) {
        /*int i = stripes - 1;
        for (int j = 0; j < commSize; ++j) {
            if (commRank == j) {
                A.readColumn(a, i);
                countPivot(a, x, K);
            }
            MPI_Bcast(&x.data[K], N - K, MPI_DOUBLE, j, MPI_COMM_WORLD);

            matMul(A, tmpA, U, x, K);
            if (commRank == j) A.writeColumn(a, i);

            K++;
        }*/

        MPI_Comm resComm, dupCommWorld;
        MPI_Comm_dup(MPI_COMM_WORLD, &dupCommWorld);

        MPI_Group gWorld, gRes;
        MPI_Comm_group(MPI_COMM_WORLD, &gWorld);

        int *ranks = new int[residual + 1];
        for (int i = 0; i < residual + 1; ++i) ranks[i] = i;

        MPI_Group_incl(gWorld, residual + 1, ranks, &gRes);
        MPI_Comm_create(dupCommWorld, gRes, &resComm);
        MPI_Group_free(&gRes); // ATTENTION

        if (commRank < residual + 1) {
            for (int j = 0; j < residual - 1; ++j) {

                if (commRank == j) {
                    A.readColumn(a, A.localM - 1);
                    countPivot(a, x, K);
                }
                MPI_Bcast(&x.data[K], N - K, MPI_DOUBLE, j, resComm);

                matMul(A, tmpA, U, x, K);
                if (commRank == j) A.writeColumn(a, A.localM - 1);

                K++;
            }
        }
        delete[] ranks;
    }
    else {
        int i = stripes - 1;
        for (int j = 0; j < commSize - 1; ++j) {
            if (commRank == j) {
                A.readColumn(a, i);
                countPivot(a, x, K);
            }
            MPI_Bcast(&x.data[K], N - K, MPI_DOUBLE, j, MPI_COMM_WORLD);

            matMul(A, tmpA, U, x, K);
            if (commRank == j) A.writeColumn(a, i);

            K++;
        }
    }
}

void solve(CycleMatrix &A, Vector &x)
{
    int N = A.size, commRank = A.commRank, commSize = A.commSize, localM = A.localM;
    int residual = N % commSize;
    // Vector x(localM);
    Vector f(N);

    int stripes = localM - 1;
    if (commRank == residual) {
        A.readColumn(f, localM - 1);
        localM -= 1;
        stripes -= 1;
    }
    MPI_Bcast(f.data, N, MPI_DOUBLE, residual, MPI_COMM_WORLD);
    double tmp = 0, rightParts[N];
    for (int i = 0; i < N; ++i) rightParts[i] = 0; // f.data[i];
    // Vector rightParts(N);

    residual = (residual == 0) ? commSize : residual;
// last stripe
    if (commRank < residual) {
        // cout << "Irek " << commRank;
        int j = localM - 1;
        rightParts[commSize * j + commRank] = f.data[commSize * j + commRank];

        for (int rank = residual - 1; rank > commRank; --rank) {
            MPI_Recv(&tmp, 1, MPI_DOUBLE, rank, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            rightParts[commSize * j + commRank] += tmp;
        }
        x.data[j] = rightParts[commSize * j + commRank] / A.data[j][commSize * j + commRank];
        // cout << x.data[j] << endl;

        for (int rank = commRank - 1; rank >= 0; --rank) {
            tmp = -A.data[j][commSize * j + rank] * x.data[j];
            MPI_Send(&tmp, 1, MPI_DOUBLE, rank, j, MPI_COMM_WORLD);
        }
        for (int rank = commSize - 1; rank > commRank; --rank) {
            tmp = -A.data[j][commSize * (j - 1) + rank] * x.data[j];
            MPI_Send(&tmp, 1, MPI_DOUBLE, rank, j - 1, MPI_COMM_WORLD);
        }
        stripes -= 1;
        for (int i = commSize * j - 1; i >= 0; --i) rightParts[i] -= A.data[j][i] * x.data[j];
    }
//
    else {
        for (int rank = commSize - 1; rank > commRank; --rank) {
            tmp = 0;
            MPI_Send(&tmp, 1, MPI_DOUBLE, rank, localM - 1, MPI_COMM_WORLD);
        }
    }

    for (int j = stripes; j >= 0; --j) {
        /*for (int rank = 0; rank < commSize; ++rank) rightParts[rank] = 0;
        rightParts[commRank] = f.data[commSize * j + commRank];
        for (int i = j + 1; i < localM; ++i) {
            for (int rank = 0; rank <= commRank; ++rank)
                rightParts[rank] -= A.data[i][commSize * j + rank] * x.data[i];

            for (int rank = commRank + 1; rank < commSize; ++rank)
                rightParts[rank] -= A.data[i][commSize * (j - 1) + rank] * x.data[i];
        }*/

        // rightParts[commSize * j + commRank] -= A.data[i][commSize * j + rank] * x.data[i];

        for (int rank = commSize - 1; rank >= 0; --rank) {
            if (rank == commRank) continue;
            MPI_Recv(&tmp, 1, MPI_DOUBLE, rank, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            rightParts[commSize * j + commRank] += tmp;
        }
        x.data[j] = (f.data[commSize * j + commRank] + rightParts[commSize * j + commRank]) / A.data[j][commSize * j + commRank];
        // cout << x.data[j] << endl;

        for (int rank = commRank - 1; rank >= 0; --rank) {
            rightParts[commSize * j + rank] -= A.data[j][commSize * j + rank] * x.data[j];
            MPI_Send(&rightParts[commSize * j + rank], 1, MPI_DOUBLE, rank, j, MPI_COMM_WORLD);
        }

        if (j > 0) {
            for (int rank = commSize - 1; rank > commRank; --rank) {
                rightParts[commSize * (j - 1) + rank] -= A.data[j][commSize * (j - 1) + rank] * x.data[j];
                MPI_Send(&rightParts[commSize * (j - 1) + rank], 1, MPI_DOUBLE, rank, j - 1, MPI_COMM_WORLD);
            }
        }

        for (int i = commSize * j - 1; i >= 0; --i) rightParts[i] -= A.data[j][i] * x.data[j];
    }
}

double residual(CycleMatrix &A, Vector &x)
{
    int N = A.size, commRank = A.commRank, commSize = A.commSize, localM = A.localM;
    int residual = N % commSize;
    Vector f(N);

    if (commRank == residual) {
        A.readColumn(f, localM - 1);
        localM -= 1;
    }
    // MPI_Bcast(f.data, N, MPI_DOUBLE, residual, MPI_COMM_WORLD);

    double resNorm = 0;
    for (int j = 0; j < N; ++j) {
        double tmp = 0, tmpRecv;
        for (int i = 0; i < localM; ++i) {
            tmp += A.data[i][j] * x.data[i];
        }
        MPI_Reduce(&tmp, &tmpRecv, 1, MPI_DOUBLE, MPI_SUM, residual, MPI_COMM_WORLD);
        if (commRank == residual) resNorm += (f.data[j] - tmpRecv) * (f.data[j] - tmpRecv);
    }

    MPI_Bcast(&resNorm, 1, MPI_DOUBLE, residual, MPI_COMM_WORLD);
    return sqrt(resNorm);
}
