/*
  drawZsCollapsed.c

  draws latent topic z's
  samp = sample matrix
  cphi = tucker decomposition tensor core tensor
  cpsi = tucker decomposition matrices
  r = restaurant lists / paths per each x
  prior = 1 or 1/L
 
 The calling syntax is:
    [samp,p] = drawZsCollapsed(samp,cphi,cpsi,path,L,options)

  Created by Adam Sandler.
*/

#include "mex.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if 0
#include <omp.h>
#endif

// forward declarations
void drawZs(double *sampIn, double *sampOut, double *p, size_t sampCols,
        size_t sampRows, double *phi, const mxArray *psi,
        double *pth, double *l, const mwSize *phiDims, int pri,
        double *cut, int topic, double *weights, double *var);
void drawZsPar(double *sampIn, double *sampOut, double *p, size_t sampCols,
        size_t sampRows, double *phi, const mxArray *psi,
        double *pth, double *l, const mwSize *phiDims, int pri,
        double *cut, int topic, double *weights, double *var);
void drawZ(int j, double *sampIn, double *sampOut, double *p,
        size_t sampCols, size_t sampRows, double *phi, const mxArray *psi,
        double *psis, double *pth, double *l, const mwSize *phiDims,
        double *a, double *cut, int topic, double *weights, int modes,
        int size, int ma, double *var);
int indices(long long int x, int m, const mwSize *dims);
void normalize(double *pdf, double sum, int size);
double variance(double *pdf, int size);
long long int multi(double *pdf, int size);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
        const mxArray *prhs[]);