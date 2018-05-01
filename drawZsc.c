/*
  drawZsc.c

  draws latent topic z's
  samp = sample matrix
  phi = tucker decomposition tensor core tensor
  psi = tucker decomposition matrices
  r = restaurant lists
 
 The calling syntax is:
    [samp,p] = drawZsc(samp,phi,psi,r)

  Created by Adam Sandler.
*/

#include "mex.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// forward declarations
void drawZsc(double *sampIn, double *sampOut, double *p, size_t sampCols,
        size_t sampRows, double *phi, double *psi1, double *psi2,
        double *r1, double *r2, const mwSize *phiDims,
        const mwSize *psi1Dims, const mwSize *psi2Dims);
void drawZ(int j, double *sampIn, double *sampOut, double *p,
        size_t sampCols, size_t sampRows, double *phi, double *psi1,
        double *psi2, double *r1, double *r2, const mwSize *phiDims,
        const mwSize *psi1Dims, const mwSize *psi2Dims);
int multi(double *pdf, int size);
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]);
void normalize(double *pdf, double sum, int size);

void drawZsc(double *sampIn, double *sampOut, double *p, size_t sampCols,
        size_t sampRows, double *phi, double *psi1, double *psi2,
        double *r1, double *r2,const mwSize *phiDims,
        const mwSize *psi1Dims, const mwSize *psi2Dims)
{
    int j;
    
    srand48(time(NULL)); // randomize seed
    
    for(j=0; j<sampRows; j++){
        drawZ(j,sampIn,sampOut,p,sampCols,sampRows,phi,psi1,psi2,r1,r2,
                phiDims,psi1Dims,psi2Dims);  
    }
    
}

void drawZ(int j, double *sampIn, double *sampOut, double *p,
        size_t sampCols, size_t sampRows, double *phi, double *psi1,
        double *psi2, double *r1, double *r2, const mwSize *phiDims,
        const mwSize *psi1Dims, const mwSize *psi2Dims)
{
    int x = sampIn[0*sampRows+j]-1; //get evidence variable
    int y1 = sampIn[1*sampRows+j]-1; //get 1st response variable
    int y2 = sampIn[2*sampRows+j]-1; // get 2nd response variable

    // initialize sampOut
    int i, k;
    for(i=0; i<sampCols; i++){
        sampOut[i*sampRows+j] = sampIn[i*sampRows+j];
    }

    // get pdf from phi and psi
    int size = phiDims[1]*phiDims[2];
    double pdf[size];
    double sum = 0;
    int index;
    for(i=0; i<phiDims[1]; i++){
        for(k=0; k<phiDims[2]; k++){
            index = i+k*phiDims[1];
            pdf[index] = phi[x+i*phiDims[0]+k*phiDims[0]*phiDims[1]];
            pdf[index] *= psi1[y1+i*psi1Dims[0]];
            pdf[index] *= psi2[y2+k*psi2Dims[0]];
            sum = sum + pdf[index];
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
        z1 = z % phiDims[1];
        z2 = z / phiDims[1];
        p[j] = pdf[z];
    }

    sampOut[3*sampRows+j] = r1[z1]; //set topic
    sampOut[4*sampRows+j] = r2[z2]; //set topic

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
    double *res1, *res2; // restaurant lists
    size_t ncols, nrows, res2Size; // number of columns of sample
    const mwSize *coreDims, *aux1Dims, *aux2Dims;
    
    /* Check number of inputs and outputs */
    if(nrhs != 4) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                          "Four inputs required.");
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
    res1 = mxGetPr(mxGetCell(prhs[3],0));
    res2 = mxGetPr(mxGetCell(prhs[3],1));
    
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)nrows,(mwSize)ncols,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(nrows,1,mxREAL);

    /* get a pointer to the real data in the output matrix */
    sOut = mxGetPr(plhs[0]);
    prob = mxGetPr(plhs[1]);
    
    /* call the computational routine */
    drawZsc(sIn,sOut,prob,ncols,nrows,core,aux1,aux2,res1,res2,coreDims,
            aux1Dims,aux2Dims);
}
