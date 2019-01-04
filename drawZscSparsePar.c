/*
  drawZsCollapsed.c

  draws latent topic z's
  samp = sample matrix
  phi = tucker decomposition tensor core tensor
  psi = tucker decomposition matrices
  r = restaurant lists / paths per each x
 
 The calling syntax is:
    [samp,p] = drawZscSparsePar(samp,phi,psi,path,L)

  Created by Adam Sandler.
*/

#include "mex.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// forward declarations
void drawZs(double *sampIn, double *sampOut, double *p, size_t sampCols,
        size_t sampRows, double *phi, double *psi1, double *psi2,
        double *pth, double *l, const mwSize *phiDims,
        const mwSize *psi1Dims, const mwSize *psi2Dims);
void drawZ(int j, double *sampIn, double *sampOut, double *p,
        size_t sampCols, size_t sampRows, double *phi, double *psi1,
        double *psi2, double *pth, double *l, const mwSize *phiDims,
        const mwSize *psi1Dims, const mwSize *psi2Dims);
int multi(double *pdf, int size);
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]);
void normalize(double *pdf, double sum, int size);

void drawZs(double *sampIn, double *sampOut, double *p, size_t sampCols,
        size_t sampRows, double *phi, double *psi1, double *psi2,
        double *pth, double *l, const mwSize *phiDims,
        const mwSize *psi1Dims, const mwSize *psi2Dims)
{
    int j;
    
    #pragma omp parallel private(j)
    {
        srand48(time(NULL)+omp_get_thread_num()); // randomize seed

        #pragma omp for
        for(j=0; j<sampRows; j++){
            drawZ(j,sampIn,sampOut,p,sampCols,sampRows,phi,psi1,psi2,pth,l,
                    phiDims,psi1Dims,psi2Dims);  
        }
    }
}

void drawZ(int j, double *sampIn, double *sampOut, double *p,
        size_t sampCols, size_t sampRows, double *phi, double *psi1,
        double *psi2, double *pth, double *l, const mwSize *phiDims,
        const mwSize *psi1Dims, const mwSize *psi2Dims)
{
    int x = sampIn[0*sampRows+j]-1; //get evidence variable
    int y1 = sampIn[1*sampRows+j]-1; //get 1st response variable
    int y2 = sampIn[2*sampRows+j]-1; // get 2nd response variable
    int z1o = sampIn[3*sampRows+j]-1; //get 1st old topic
    int z2o = sampIn[4*sampRows+j]-1; // get 2nd old topic
    
    // initialize sampOut
    int i, k;
    for(i=0; i<sampCols; i++){
        sampOut[i*sampRows+j] = sampIn[i*sampRows+j];
    }
    
    // get pdf from phi and psi
    int size = floor(l[0]*l[1]);
    double pdf[size];
    double sum = 0;
    int ind;
    int ip, kp, indp;
    int l0 = floor(l[0]);
    int l1 = floor(l[1]);
    for(i=0; i<l0; i++){
        for(k=0; k<l1; k++){
            ip = pth[x+i*phiDims[0]]-1;
            kp = pth[x+(k+l0)*phiDims[0]]-1;
            ind = i+k*l0;
            if(phiDims[1]==l0){
                indp = ind;
            } else{
                indp = ip+kp*phiDims[1];
            }
            pdf[ind] = phi[x+indp*phiDims[0]];
            pdf[ind] *= psi1[y1+ip*psi1Dims[0]];
            pdf[ind] *= psi2[y2+kp*psi2Dims[0]];
            sum = sum + pdf[ind];
            if(sum >= DBL_MAX) {
                mexErrMsgIdAndTxt("MyProg:sum:overflow",
                                  "Sum overflow.");
            }
        }
    }
    
    // draw new z
    int z, z1, z2;
    if(sum==0){
        z1 = 0;
        z2 = 0;
        p[j] = 1.0;
    } else{
        normalize(pdf,sum,size);
        z = multi(pdf,size);
        z1 = z % l0;
        z2 = z / l0;
        p[j] = pdf[z];
    }

    sampOut[3*sampRows+j] = pth[x+z1*phiDims[0]]; //set topic
    sampOut[4*sampRows+j] = pth[x+(z2+l0)*phiDims[0]]; //set topic

}

//normalizes pdf
void normalize(double *pdf, double sum, int size){
    int i;
    
    for(i=0; i<size; i++){
        pdf[i] = pdf[i]/sum;
    }
}

/* generates single value from multinomial pdf
  pdf = vector of probabilities
  x = sample */
int multi(double *pdf, int size){
    double cdf[size];
    int i;
    
    // compute cdf
    cdf[0]=pdf[0];
    for(i=1; i<size; i++){
        cdf[i]=cdf[i-1]+pdf[i];
    }
    
    double n = 0;
    n = drand48(); // get uniform random variable
    
    // find bin that n is in
    i = 0;
    int found = 0;
    while(i<size && found==0){
        if(cdf[i]>n){
            found = 1;
        } else{
            i++;
        }
    }
    
    //free(cdf);
    return i;
}

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    /* variable declarations */
    double *sIn, *sOut, *prob; // sample row
    double *core; // tucker decomposition tensor core tensor
    double *aux1, *aux2; // tucker decomposition matrices
    double *path; // pathways
    double *L; // levels
    size_t ncols, nrows, res2Size; // number of columns of sample
    const mwSize *coreDims, *aux1Dims, *aux2Dims;
    
    /* Check number of inputs and outputs */
    if(nrhs != 5) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                          "Five inputs required.");
    }
    if(nlhs != 2) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs",
                          "Two outputs required.");
    }
    
    sIn = mxGetPr(prhs[0]);
    ncols = mxGetN(prhs[0]);
    nrows = mxGetM(prhs[0]);
    core = mxGetPr(prhs[1]);
    coreDims = mxGetDimensions(prhs[1]);
    aux1 = mxGetPr(mxGetCell(prhs[2],0));
    aux1Dims = mxGetDimensions(mxGetCell(prhs[2],0));
    aux2 = mxGetPr(mxGetCell(prhs[2],1));
    aux2Dims = mxGetDimensions(mxGetCell(prhs[2],1));
    path = mxGetPr(prhs[3]);
    L = mxGetPr(prhs[4]);
    
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)nrows,(mwSize)ncols,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(nrows,1,mxREAL);

    /* get a pointer to the real data in the output matrix */
    sOut = mxGetPr(plhs[0]);
    prob = mxGetPr(plhs[1]);
    
    /* call the computational routine */
    drawZs(sIn,sOut,prob,ncols,nrows,core,aux1,aux2,path,L,coreDims,
            aux1Dims,aux2Dims);
}
