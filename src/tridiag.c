#include "eigensolver.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double pythag(double a, double b)
{
    double absa = fabs(a), absb = fabs(b);
    if (absa > absb) {
        double r = absb / absa;
        return absa * sqrt(1.0 + r * r);
    }
    if (absb == 0.0) return 0.0;
    double r = absa / absb;
    return absb * sqrt(1.0 + r * r);
}

static int tql2_eigvals(int n, double *d, double *e)
{
    if (n == 1) return 0;

    for (int i = 1; i < n; ++i) e[i - 1] = e[i];
    e[n - 1] = 0.0;

    for (int l = 0; l < n; ++l) {
        int iter = 0;
        int m;
        do {
            for (m = l; m < n - 1; ++m) {
                double dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m]) + dd == dd) break;
            }
            if (m != l) {
                if (iter++ == 60) return -1;
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = pythag(g, 1.0);
                double sgn = (g >= 0.0) ? fabs(r) : -fabs(r);
                g = d[m] - d[l] + e[l] / (g + sgn);
                double s = 1.0, c = 1.0, p = 0.0;
                int i;
                for (i = m - 1; i >= l; --i) {
                    double f = s * e[i];
                    double b = c * e[i];
                    r = pythag(f, g);
                    e[i + 1] = r;
                    if (r == 0.0) {
                        d[i + 1] -= p;
                        e[m] = 0.0;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - b;
                }
                if (r == 0.0 && i >= l) continue;
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }

    for (int i = 0; i < n - 1; ++i) {
        int k = i;
        double p = d[i];
        for (int j = i + 1; j < n; ++j)
            if (d[j] < p) { k = j; p = d[j]; }
        if (k != i) { d[k] = d[i]; d[i] = p; }
    }
    return 0;
}

int tridiag_eigenvalues(const double *alpha, const double *beta,
                        int m, double *eigvals)
{
    double *d = (double *)malloc(m * sizeof(double));
    double *e = (double *)malloc(m * sizeof(double));
    if (!d || !e) { free(d); free(e); return -1; }
    memcpy(d, alpha, m * sizeof(double));
    e[0] = 0.0;
    for (int i = 1; i < m; ++i) e[i] = beta[i - 1];
    int rc = tql2_eigvals(m, d, e);
    if (rc == 0) memcpy(eigvals, d, m * sizeof(double));
    free(d); free(e);
    return rc;
}

static double sign_d(double a, double b) {
    return (b >= 0.0) ? fabs(a) : -fabs(a);
}

int hessenberg_eigenvalues(const double *Hin, int m,
                           double *wr, double *wi)
{
    if (m <= 0) return -1;
    double *H = (double *)malloc((size_t)m * m * sizeof(double));
    if (!H) return -1;
    memcpy(H, Hin, (size_t)m * m * sizeof(double));

    double anorm = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = (i > 0 ? i - 1 : 0); j < m; ++j)
            anorm += fabs(H[i * m + j]);
    }

    int nn = m - 1;
    double t = 0.0;
    while (nn >= 0) {
        int its = 0;
        int l;
        do {
            for (l = nn; l >= 1; --l) {
                double s = fabs(H[(l - 1) * m + (l - 1)]) +
                           fabs(H[l * m + l]);
                if (s == 0.0) s = anorm;
                if (fabs(H[l * m + (l - 1)]) + s == s) {
                    H[l * m + (l - 1)] = 0.0;
                    break;
                }
            }
            double x = H[nn * m + nn];
            if (l == nn) {
                wr[nn] = x + t;
                wi[nn] = 0.0;
                --nn;
            } else {
                double y = H[(nn - 1) * m + (nn - 1)];
                double w = H[nn * m + (nn - 1)] * H[(nn - 1) * m + nn];
                if (l == nn - 1) {
                    double p = 0.5 * (y - x);
                    double q = p * p + w;
                    double z = sqrt(fabs(q));
                    x += t;
                    if (q >= 0.0) {
                        z = p + sign_d(z, p);
                        wr[nn - 1] = x + z;
                        wr[nn] = (z != 0.0) ? (x - w / z) : wr[nn - 1];
                        wi[nn - 1] = 0.0;
                        wi[nn] = 0.0;
                    } else {
                        wr[nn - 1] = x + p;
                        wr[nn] = x + p;
                        wi[nn - 1] =  z;
                        wi[nn] = -z;
                    }
                    nn -= 2;
                } else {
                    if (its == 30) { free(H); return -1; }
                    if (its == 10 || its == 20) {
                        t += x;
                        for (int i = 0; i <= nn; ++i)
                            H[i * m + i] -= x;
                        double s = fabs(H[nn * m + (nn - 1)]) +
                                   fabs(H[(nn - 1) * m + (nn - 2)]);
                        y = x = 0.75 * s;
                        w = -0.4375 * s * s;
                    }
                    ++its;
                    int mloc;
                    double p = 0.0, q = 0.0, r = 0.0;
                    for (mloc = nn - 2; mloc >= l; --mloc) {
                        double zv = H[mloc * m + mloc];
                        r = x - zv;
                        double sl = y - zv;
                        p = (r * sl - w) / H[(mloc + 1) * m + mloc] +
                            H[mloc * m + (mloc + 1)];
                        q = H[(mloc + 1) * m + (mloc + 1)] - zv - r - sl;
                        r = H[(mloc + 2) * m + (mloc + 1)];
                        double s = fabs(p) + fabs(q) + fabs(r);
                        p /= s; q /= s; r /= s;
                        if (mloc == l) break;
                        double u = fabs(H[mloc * m + (mloc - 1)]) *
                                   (fabs(q) + fabs(r));
                        double v = fabs(p) *
                                   (fabs(H[(mloc - 1) * m + (mloc - 1)]) +
                                    fabs(zv) +
                                    fabs(H[(mloc + 1) * m + (mloc + 1)]));
                        if (u + v == v) break;
                    }
                    for (int i = mloc + 2; i <= nn; ++i) {
                        H[i * m + (i - 2)] = 0.0;
                        if (i != mloc + 2) H[i * m + (i - 3)] = 0.0;
                    }
                    for (int k = mloc; k <= nn - 1; ++k) {
                        if (k != mloc) {
                            p = H[k * m + (k - 1)];
                            q = H[(k + 1) * m + (k - 1)];
                            r = (k != nn - 1) ? H[(k + 2) * m + (k - 1)] : 0.0;
                            x = fabs(p) + fabs(q) + fabs(r);
                            if (x == 0.0) continue;
                            p /= x; q /= x; r /= x;
                        }
                        double s = sign_d(sqrt(p * p + q * q + r * r), p);
                        if (k == mloc) {
                            if (l != mloc)
                                H[k * m + (k - 1)] = -H[k * m + (k - 1)];
                        } else {
                            H[k * m + (k - 1)] = -s * x;
                        }
                        p += s;
                        x = p / s;
                        double yv = q / s;
                        double zv = r / s;
                        q /= p; r /= p;
                        for (int j = k; j <= nn; ++j) {
                            p = H[k * m + j] + q * H[(k + 1) * m + j];
                            if (k != nn - 1) {
                                p += r * H[(k + 2) * m + j];
                                H[(k + 2) * m + j] -= p * zv;
                            }
                            H[(k + 1) * m + j] -= p * yv;
                            H[k * m + j] -= p * x;
                        }
                        int mmin = (nn < k + 3) ? nn : k + 3;
                        for (int i = l; i <= mmin; ++i) {
                            p = x * H[i * m + k] + yv * H[i * m + (k + 1)];
                            if (k != nn - 1) {
                                p += zv * H[i * m + (k + 2)];
                                H[i * m + (k + 2)] -= p * r;
                            }
                            H[i * m + (k + 1)] -= p * q;
                            H[i * m + k] -= p;
                        }
                    }
                }
            }
        } while (l < nn - 1);
    }
    free(H);
    return 0;
}
