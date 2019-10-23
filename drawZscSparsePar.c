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
        size_t sampRows, double *phi, const mxArray *psi, double *pth,
        double *l, const mwSize *phiDims);
void drawZ(int j, double *sampIn, double *sampOut, double *p,
        size_t sampCols, size_t sampRows, double *phi, const mxArray *psi,
        double *pth, double *l, const mwSize *phiDims);
int indices(long long int x, long long int m, const mwSize *dims);
void normalize(double *pdf, double sum, int size);
long long int multi(double *pdf, int size);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
        const mxArray *prhs[]);

void drawZs(double *sampIn, double *sampOut, double *p, size_t sampCols,
        size_t sampRows, double *phi, const mxArray *psi, double *pth,
        double *l, const mwSize *phiDims)
{
    int j;
    
    #pragma omp parallel private(j)
    {
        srand48(time(NULL)+omp_get_thread_num()); // randomize seed

        #pragma omp for
        for(j=0; j<sampRows; j++){
            drawZ(j,sampIn,sampOut,p,sampCols,sampRows,phi,psi,pth,l,
                    phiDims); 
        }
    }
}

void drawZ(int j, double *sampIn, double *sampOut, double *p,
        size_t sampCols, size_t sampRows, double *phi, const mxArray *psi,
        double *pth, double *l, const mwSize *phiDims)
{
    int x = sampIn[0*sampRows+j]-1; //get evidence variable
    int y; //response variable
    
    // initialize sampOut
    long long int i; int k;
    for(i=0; i<sampCols; i++){
        sampOut[i*sampRows+j] = sampIn[i*sampRows+j];
    }
    int modes = (sampCols-1)/2;
    
    // check dims
    for(k=0; k<modes; k++){
        if(phiDims[k+1]!=l[k]){
            mexErrMsgIdAndTxt("MyProg:dims:mismatch",
                "Dims of phi don't match number of levels.");
        }
    }
    
    // get pdf from phi and psi
    long long int size=1;
    for(i=0; i<modes; i++){
        size *= floor(l[i]);
    }
    double pdf[size]; double pdf1; double sum = 0; int ind, ip, lsum;
    double *psik; const mwSize *psikDims;
    for(i=0; i<size; i++){
        pdf1 = log(phi[x+i*phiDims[0]]);
        lsum = 0;
        for(k=0; k<modes; k++){
            ind = indices(i,k,&phiDims[1]);
            ip = pth[x+(ind+lsum)*phiDims[0]]-1;
            lsum += l[k];
            y = sampIn[(k+1)*sampRows+j]-1;
            psik = mxGetPr(mxGetCell(psi,k));
            psikDims = mxGetDimensions(mxGetCell(psi,k));
            pdf1 += log(psik[y+ip*psikDims[0]]);
        }
        pdf[i] = exp(pdf1);
        sum = sum + pdf[i];
        if(sum >= DBL_MAX) {
            mexErrMsgIdAndTxt("MyProg:sum:overflow", "Sum overflow.");
        }
    }
    
    // draw new z
    long long int z; int z1;
    if(sum==0){
        z = 0;
        p[j] = 1.0;
    } else{
        normalize(pdf,sum,size);
        z = multi(pdf,size);
        p[j] = pdf[z];
    }
    lsum = 0;
    for(k=0; k<modes; k++){
        z1 = indices(z,k,&phiDims[1]);
        //set topic
        sampOut[(1+modes+k)*sampRows+j] = pth[x+(z1+lsum)*phiDims[0]];
        lsum += l[k];
    }

}

int indices(long long int x, long long int m, const mwSize *dims){
    long long int t=1; int i, o;
    if(m>0){
        for(i=0; i<m; i++){
            t *= dims[i];
        }
    }
    o = floor(x/t);
    t *= dims[m];
    o = o % t;
    return o;
}

//normalizes pdf
void normalize(double *pdf, double sum, int size){
    long long int i;
    for(i=0; i<size; i++){
        pdf[i] = pdf[i]/sum;
    }
}

/* generates single value from multinomial pdf
  pdf = vector of probabilities
  x = sample */
long long int multi(double *pdf, int size){
    double cdf[size];
    long long int i;
    
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
    const mxArray *aux; // tucker decomposition matrices
    double *path; // pathways
    double *L; // levels
    size_t ncols, nrows; // number of columns of sample
    const mwSize *coreDims;
    
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
    aux = prhs[2];
    path = mxGetPr(prhs[3]);
    L = mxGetPr(prhs[4]);
    
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)nrows,(mwSize)ncols,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(nrows,1,mxREAL);

    /* get a pointer to the real data in the output matrix */
    sOut = mxGetPr(plhs[0]);
    prob = mxGetPr(plhs[1]);
    
    /* call the computational routine */
    drawZs(sIn,sOut,prob,ncols,nrows,core,aux,path,L,coreDims);
}
