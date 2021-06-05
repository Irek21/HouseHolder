void countPivot(Vector &a, Vector &x, int K);
void matMul(CycleMatrix &A, CycleMatrix &tmpA, Vector &U, Vector &x, int K);
void householder(CycleMatrix &A);
void solve(CycleMatrix &A, Vector &x);
double residual(CycleMatrix &A, Vector &x);
