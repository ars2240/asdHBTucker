/*
  drawZsCollapsed.c

  draws latent topic z's
  samp = sample matrix
  cphi = tucker decomposition tensor core tensor
  cpsi = tucker decomposition matrices
  r = restaurant lists
  prior = 1 or 1/L
 
 The calling syntax is:
    [samp,p] = drawZsCollapsed(samp,cphi,cpsi,r,prior)

  Created by Adam Sandler.
*/

#include "mex.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// forward declarations
void drawZs(double *sampIn, double *sampOut, double *p, size_t sampCols,
        size_t sampRows, double *phi, double *psi1, double *psi2,
        double *r1, double *r2, const mwSize *phiDims,
        const mwSize *psi1Dims, const mwSize *psi2Dims, int pri);
void drawZ(int j, double *sampIn, double *sampOut, double *p,
        size_t sampCols, size_t sampRows, double *phi, double *psi1,
        double *psi2, double *psi1s, double *psi2s, double *r1, double *r2,
        const mwSize *phiDims, const mwSize *psi1Dims, 
        const mwSize *psi2Dims, double a1, double a2, double a3);
int multi(double *pdf, int size);
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]);
void normalize(double *pdf, double sum, int size);

void drawZs(double *sampIn, double *sampOut, double *p, size_t sampCols,
        size_t sampRows, double *phi, double *psi1, double *psi2,
        double *r1, double *r2, const mwSize *phiDims,
        const mwSize *psi1Dims, const mwSize *psi2Dims, int pri)
{
    int j;
    
    // setup alphas
    double alpha1, alpha2, alpha3;
    if(pri==1){
        alpha1=1.0;
        alpha2=1.0;
        alpha3=1.0;
    } else{
        alpha1=1/phiDims[1]/phiDims[2];
        alpha2=1/psi1Dims[0];
        alpha3=1/psi2Dims[0];
    }
    
    double psi1Sum[psi1Dims[1]];
    double psi2Sum[psi2Dims[1]];
    int i, k;
    for(i=0; i<psi1Dims[1]; i++){
        psi1Sum[i]=alpha2*psi1Dims[0];
        for(k=0; k<psi1Dims[0]; k++){
            psi1Sum[i]+=psi1[k+i*psi1Dims[0]];
        }
        mexPrintf("Psi1 - topic %d of %d, sum = %f.\n",i+1,psi1Dims[1],psi1Sum[i]);
    }
    for(i=0; i<psi2Dims[1]; i++){
        psi2Sum[i]=alpha3*psi2Dims[0];
        for(k=0; k<psi2Dims[0]; k++){
            psi2Sum[i]+=psi2[k+i*psi2Dims[0]];
        }
        mexPrintf("Psi2 - topic %d of %d, sum = %f.\n",i+1,psi2Dims[1],psi2Sum[i]);
    }
    
    srand48(time(NULL)); // randomize seed
    
    mexPrintf("Initialized.\n");
    for(j=0; j<sampRows; j++){
        drawZ(j,sampIn,sampOut,p,sampCols,sampRows,phi,psi1,psi2,psi1Sum,
                psi2Sum,r1,r2,phiDims,psi1Dims,psi2Dims,alpha1,alpha2,
                alpha3);  
    }
    
}

void drawZ(int j, double *sampIn, double *sampOut, double *p,
        size_t sampCols, size_t sampRows, double *phi, double *psi1,
        double *psi2, double *psi1s, double *psi2s, double *r1, double *r2,
        const mwSize *phiDims, const mwSize *psi1Dims, 
        const mwSize *psi2Dims, double a1, double a2, double a3)
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
            pdf[index] = phi[x+index*phiDims[0]] + a1;
            pdf[index] *= (psi1[y1+i*psi1Dims[0]] + a2)/psi1s[i];
            pdf[index] *= (psi2[y2+k*psi2Dims[0]] + a3)/psi2s[k];
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
    int prior; // prior
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
    res1 = mxGetPr(mxGetCell(prhs[3],0));
    res2 = mxGetPr(mxGetCell(prhs[3],1));
    prior = (int)*mxGetPr(prhs[4]);
    
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)nrows,(mwSize)ncols,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(nrows,1,mxREAL);

    /* get a pointer to the real data in the output matrix */
    sOut = mxGetPr(plhs[0]);
    prob = mxGetPr(plhs[1]);
    
    /* call the computational routine */
    drawZs(sIn,sOut,prob,ncols,nrows,core,aux1,aux2,res1,res2,coreDims,
            aux1Dims,aux2Dims,prior);
}
