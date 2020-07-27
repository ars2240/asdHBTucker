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
        size_t sampRows, double *phi, const mxArray *psi, const mxArray *r,
        const mwSize *phiDims);
void drawZ(int j, double *sampIn, double *sampOut, double *p,
        size_t sampCols, size_t sampRows, double *phi, const mxArray *psi,
        const mxArray *r, const mwSize *phiDims);
int indices(long long int x, int m, const mwSize *dims);
long long int multi(double *pdf, int size);
void normalize(double *pdf, double sum, int size);
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]);

void drawZsc(double *sampIn, double *sampOut, double *p, size_t sampCols,
        size_t sampRows, double *phi, const mxArray *psi, const mxArray *r,
        const mwSize *phiDims)
{
    int j;
    
    srand48(time(NULL)); // randomize seed
    
    for(j=0; j<sampRows; j++){
        drawZ(j,sampIn,sampOut,p,sampCols,sampRows,phi,psi,r,phiDims);  
    }
    
}

void drawZ(int j, double *sampIn, double *sampOut, double *p,
        size_t sampCols, size_t sampRows, double *phi, const mxArray *psi,
        const mxArray *r, const mwSize *phiDims)
{
    int x = sampIn[0*sampRows+j]-1; //get evidence variable
    int y; //response variable

    // initialize sampOut
    long long int i; int k;
    for(i=0; i<sampCols; i++){
        sampOut[i*sampRows+j] = sampIn[i*sampRows+j];
    }
    int modes = (sampCols-1)/2;

    // get pdf from phi and psi
    long long int size=1;
    for(i=1; i<=modes; i++){
        size *= phiDims[i];
    }
    double pdf[size]; double pdf1; double sum = 0; int index, ind;
    double *psik; const mwSize *psikDims;
    for(i=0; i<size; i++){
        pdf1 = log(phi[x+i*phiDims[0]]);
        // mexPrintf("phi[%d,%d] = %.3E\n", x+1, i+1, phi[x+i*phiDims[0]]);
        for(k=0; k<modes; k++){
            ind = indices(i,k,&phiDims[1]);
            psik = mxGetPr(mxGetCell(psi,k));
            psikDims = mxGetDimensions(mxGetCell(psi,k));
            y = sampIn[(k+1)*sampRows+j]-1;
            pdf1 += log(psik[y+ind*psikDims[0]]);
            // mexPrintf("psi{%d}[%d,%d] = %.3E\n", k+1, y+1, ind+1, psik[y+ind*psikDims[0]]);
        }
        pdf[i] = exp(pdf1);
        sum = sum + pdf[i];
        if(sum >= DBL_MAX) {
            mexErrMsgIdAndTxt("MyProg:sum:overflow", "Sum overflow.");
        }
    }
    // mexPrintf("sum = %.3E\n", sum);

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
    double *r1;
    for(k=0; k<modes; k++){
        z1 = indices(z,k,&phiDims[1]);
        r1 = mxGetPr(mxGetCell(r,k));
        sampOut[(1+modes+k)*sampRows+j] = round(r1[z1]); //set topic
        if(round(r1[z1]) ==0){
            mexPrintf("k = %d\n", k);
            mexPrintf("z = %d\n", z);
            mexPrintf("z1 = %d\n", z1);
            mexErrMsgIdAndTxt("MyProg:badVal:badTopic", "Bad Topic.");
        }
    }

}

int indices(long long int x, int m, const mwSize *dims){
    int t=1; int i, o;
    for(i=0; i<=m; i++){
        t *= dims[i];
    }
    o = x % t;
    t /= dims[m];
    o = floor(o/t);
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
    const mxArray *res; // restaurant lists
    size_t ncols, nrows; // number of columns of sample
    const mwSize *coreDims;
    
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
    aux = prhs[2];
    res = prhs[3];
    
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)nrows,(mwSize)ncols,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(nrows,1,mxREAL);

    /* get a pointer to the real data in the output matrix */
    sOut = mxGetPr(plhs[0]);
    prob = mxGetPr(plhs[1]);
    
    /* call the computational routine */
    drawZsc(sIn,sOut,prob,ncols,nrows,core,aux,res,coreDims);
}
