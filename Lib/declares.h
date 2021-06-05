#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <time.h>
using namespace std;

class Vector {
public:
    double *data;
    int size;

    Vector(int N) {
        size = N;
        data = new double[N];
        if (data == NULL) {
            cerr << "Error on matrix allocation" << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    void print() {
        cout << "[";
        for (int i = 0; i < size - 1; ++i) {
            cout << setw(10) << data[i] << ",";
        }
        cout << setw(10) << data[size - 1] << "],";
        cout << endl;
    }

    ~Vector() {
        delete[] data;
    }
};

class CycleMatrix {
public:
    double **data;
    int size;
    int localM;
    int commRank;
    int commSize;
    int residual;

    CycleMatrix(int N, int mpiCommRank, int mpiCommSize) {
        commRank = mpiCommRank;
        commSize = mpiCommSize;
        int stripes = (N + 1) / commSize; residual = (N + 1) % commSize; // N + 1 !
        // residual = (residual == 0) ? commSize : residual;
        localM = stripes;
        if (commRank < residual) localM++;

        size = N;
        data = new double*[localM];
        if (data == NULL) {
            cerr << "Error on matrix allocation" << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        for (int i = 0; i < localM; ++i) {
            data[i] = new double[N];
            if (data[i] == NULL) {
                cerr << "Error on matrix allocation" << endl;
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
    }

    void gen() {
        /*for (int i = 0; i < localM; ++i) {
            for (int j = 0; j < size; ++j) {
                data[i][j] = rand() % 100;
            }
        }*/

        double reduct[size], reductRecv[size];
        for (int j = 0; j < size; ++j) {
            reduct[j] = 0;
            reductRecv[j] = 0;
        }

        int iA = 0, stripes = localM;
        if (commRank == size % commSize) stripes = localM - 1;
        for (int i = 0; i < stripes; ++i) {
            for (int j = 0; j < size; ++j) {
                // data[i][j] = rand() % 100;
                // data[i][j] = (iA + 1) * (iA + 1) ;

                iA = i * commSize + commRank;
                data[i][j] = 1.0 / (iA + j + 1);
                if (iA % 2 == 0) reduct[j] += data[i][j];
            }
        }

        // for (int j = 0; j < size; ++j) {
            MPI_Reduce(reduct, reductRecv, size, MPI_DOUBLE, MPI_SUM, size % commSize, MPI_COMM_WORLD);
        // }
        if (commRank == size % commSize) {
            for (int j = 0; j < size; ++j) {
                data[localM - 1][j] = reductRecv[j];
            }
        }
    }

    void readColumn(Vector &a, int i) {
        for (int j = 0; j < size; ++j) {
            a.data[j] = data[i][j];
        }
    }

    void writeColumn(Vector &a, int i) {
        for (int j = 0; j < size; ++j) {
            data[i][j] = a.data[j];
        }
    }

    void print() {
        Vector a(size);
        int stripes = (size + 1) / commSize;
        for (int i = 0; i < stripes; ++i) {
            for (int j = 0; j < commSize; ++j) {
                if (commRank == j) {
                    readColumn(a, i);
                    a.print();
                    cout << flush;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }
        for (int j = 0; j < residual; ++j) {
            if (commRank == j) {
                readColumn(a, localM - 1);
                a.print();
                cout << flush;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    ~CycleMatrix() {
        for (int i = 0; i < localM; ++i) {
            delete[] data[i];
        }
        delete[] data;
    }
};
