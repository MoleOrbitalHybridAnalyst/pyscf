#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "config.h"
#include "cint.h"
#include "pbc/cell.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define RCUT_EPS 1e-3

double pgf_rcut(int l, double alpha, double coeff, 
                double precision, double r0, int max_cycle)
{
    double rcut;
    if (l == 0) {
        if (coeff <= precision) {
            rcut = 0;
        }
        else {
            rcut = sqrt(log(coeff / precision) / alpha);
        }
        return rcut;
    }

    double rmin = sqrt(.5 * l / alpha);
    double gmax = coeff * pow(rmin, l) * exp(-alpha * rmin * rmin);
    if (gmax < precision) {
        return 0;
    }

    int i;
    double eps = MIN(rmin/10, RCUT_EPS);
    double log_c_by_prec= log(coeff / precision);
    double rcut_old;
    rcut = MAX(r0, rmin+eps);
    for (i = 0; i < max_cycle; i++) {
        rcut_old = rcut;
        rcut = sqrt((l*log(rcut) + log_c_by_prec) / alpha);
        if (fabs(rcut - rcut_old) < eps) {
            break;
        }
    }
    if (i == max_cycle) {
        printf("pgf_rcut did not converge");
    }
    return rcut; 
}

void rcut_by_shells(double* shell_radius, double** ptr_pgf_rcut, 
                    int* bas, double* env, int nbas, 
                    double r0, double precision)
{
    int max_cycle = RCUT_MAX_CYCLE;
#pragma omp parallel
{
    int ib, ic, p;
    #pragma omp for schedule(static)
    for (ib = 0; ib < nbas; ib ++) {
        int l = bas[ANG_OF+ib*BAS_SLOTS];
        int nprim = bas[NPRIM_OF+ib*BAS_SLOTS];
        int ptr_exp = bas[PTR_EXP+ib*BAS_SLOTS];
        int nctr = bas[NCTR_OF+ib*BAS_SLOTS];
        int ptr_c = bas[PTR_COEFF+ib*BAS_SLOTS];
        double rcut_max = 0, rcut;
        for (p = 0; p < nprim; p++) {
            double alpha = env[ptr_exp+p];
            double cmax = 0;
            for (ic = 0; ic < nctr; ic++) {
                cmax = MAX(fabs(env[ptr_c+ic*nprim+p]), cmax);
            }
            rcut = pgf_rcut(l, alpha, cmax, precision, r0, max_cycle);
            if (ptr_pgf_rcut) {
                ptr_pgf_rcut[ib][p] = rcut;
            }
            rcut_max = MAX(rcut, rcut_max);
        }
        shell_radius[ib] = rcut_max;
    }
}
}

void get_SI(complex double* out, double* coords, double* Gv, int natm, int ngrid)
{
#pragma omp parallel
{
    int i, ia;
    double RG;
    double *pcoords, *pGv;
    complex double *pout;
    #pragma omp for schedule(static)
    for (ia = 0; ia < natm; ia++) {
        pcoords = coords + ia * 3;
        pout = out + ((size_t)ia) * ngrid;
        for (i = 0; i < ngrid; i++) {
            pGv = Gv + i * 3;
            RG = pcoords[0] * pGv[0] + pcoords[1] * pGv[1] + pcoords[2] * pGv[2];
            pout[i] = cos(RG) - _Complex_I * sin(RG);
        }
    }
}
}


void get_Gv(double* Gv, double* rx, double* ry, double* rz, int* mesh, double* b)
{
#pragma omp parallel
{
    int x, y, z;
    double *pGv;
    #pragma omp for schedule(dynamic)
    for (x = 0; x < mesh[0]; x++) {
        pGv = Gv + x * (size_t)mesh[1] * mesh[2] * 3;
        for (y = 0; y < mesh[1]; y++) {
        for (z = 0; z < mesh[2]; z++) {
            pGv[0]  = rx[x] * b[0];
            pGv[0] += ry[y] * b[3];
            pGv[0] += rz[z] * b[6];
            pGv[1]  = rx[x] * b[1];
            pGv[1] += ry[y] * b[4];
            pGv[1] += rz[z] * b[7];
            pGv[2]  = rx[x] * b[2];
            pGv[2] += ry[y] * b[5];
            pGv[2] += rz[z] * b[8];
            pGv += 3;
        }}
    }
}
}

void contract_rhoG_Gv(double complex* out, double complex* rhoG, double* Gv,
                      int ndens, size_t ngrids)
{
    int i;
    double complex *outx, *outy, *outz;
    for (i = 0; i < ndens; i++) {
        outx = out;
        outy = outx + ngrids;
        outz = outy + ngrids;
#pragma omp parallel
{
        size_t igrid;
        double *pGv;
        #pragma omp for schedule(static)
        for (igrid = 0; igrid < ngrids; igrid++) {
            pGv = Gv + igrid * 3;
            outx[igrid] = pGv[0] * creal(rhoG[igrid]) * _Complex_I - pGv[0] * cimag(rhoG[igrid]);
            outy[igrid] = pGv[1] * creal(rhoG[igrid]) * _Complex_I - pGv[1] * cimag(rhoG[igrid]);
            outz[igrid] = pGv[2] * creal(rhoG[igrid]) * _Complex_I - pGv[2] * cimag(rhoG[igrid]);
        }
}
        rhoG += ngrids;
        out += 3 * ngrids;
    }
}
