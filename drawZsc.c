/*
  drawZsc.c

  draws latent topic z's
  samp = sample matrix
  phi = tucker decomposition tensor core tensor
  psi = tucker decomposition matrices
  r = restaurant lists
 
 The calling syntax is:
    samp = drawZsc(samp,phi,psi,r)

  Created by Adam Sandler.
*/

#include "mex.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

void drawZsc(double *sampIn, double *sampOut, size_t sampCols,
        size_t sampRows, double *phi, double *psi1, double *psi2,
        double *r1, double *r2, size_t r1Size,const mwSize *phiDims,
        const mwSize *psi1Dims, const mwSize *psi2Dims);
double get_random();
int multi(double *pdf, double sum, int size);
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]);
/* The computational routine */
void drawZsc(double *sampIn, double *sampOut, size_t sampCols,
        size_t sampRows, double *phi, double *psi1, double *psi2,
        double *r1, double *r2, size_t r1Size,const mwSize *phiDims,
        const mwSize *psi1Dims, const mwSize *psi2Dims)
{
    int j;
    for(j=0; j<sampRows; j++){
        int x = sampIn[0*sampRows+j]; //get evidence variable
        int y = sampIn[1*sampRows+j]; //get response variable
        int z = sampIn[4*sampRows+j]; //get other topic

        // initialize sampOut
        mwSize i;
        for(i=0; i<sampCols; i++){
            sampOut[i*sampRows+j] = sampIn[i*sampRows+j];
        }

        // find z in restaurant list
        i = 0;
        int found = 0;
        while(i<r1Size && found==0){
            if(r1[i]==z){
                found = 1;
                z = i+1;
            } else{
                i++;
            }
        }
        if(found==0){
            z = 1;
        }

        // get pdf from phi and psi
        int size = phiDims[1];
        double pdf1[size];
        double sum = 0;
        for(i=0; i<size; i++){
            pdf1[i] = phi[x-1+i*phiDims[0]+(z-1)*phiDims[0]*phiDims[1]];
            pdf1[i] = psi1[y-1+i*psi1Dims[0]]*pdf1[i];
            sum = sum + pdf1[i];
        }

        // draw new z
        if(sum==0){
            z = 1;
        } else{
            z = multi(pdf1,sum,size);
        }
        //free(pdf1);

        sampOut[3*sampRows+j] = r1[z-1]; //set topic

        // sample other z
        y = sampIn[2*sampRows+j]; // get response variable

        // get pdf from phi and psi
        size = phiDims[2];
        double pdf2[size];
        sum = 0;
        for(i=0; i<size; i++){
            pdf2[i] = phi[x-1+(z-1)*phiDims[0]+i*phiDims[0]*phiDims[1]];
            pdf2[i] = psi2[y-1+i*psi2Dims[0]]*pdf2[i];
            sum = sum + pdf2[i];
        }

        // draw new z
        if(sum==0){
            z = 1;
        } else{
            z = multi(pdf2,sum,size);
        }
        //free(pdf2);

        sampOut[4*sampRows+j] = r2[z-1]; //set topic   
    }
}

// generate random uniform number
double get_random() {
    srand((int)time(NULL));
    return ((double)rand() / (double)RAND_MAX);
}

/* generates single value from multinomial pdf
  pdf = vector of probabilities
  x = sample */
int multi(double *pdf, double sum, int size){
    double cdf[size];
    int x, i;
    
    // normalize
    for(i=0; i<size; i++){
        pdf[i] = pdf[i]/sum;
    }
    
    // compute cdf
    cdf[0]=pdf[0];
    for(i=1; i<size; i++){
        cdf[i]=cdf[i-1]+pdf[i];
    }
    
    double n = 0;
    srand(time(NULL)); // randomize seed
    n = get_random(); // get uniform random variable
    
    // find bin that n is in
    i = 0;
    int found = 0;
    while(i<size && found==0){
        if(cdf[i]>n){
            found = 1;
            x = i+1;
        } else{
            i++;
        }
    }
    
    //free(cdf);
    return x;
}

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    /* variable declarations */
    double *sIn, *sOut; // sample row
    double *core; // tucker decomposition tensor core tensor
    double *aux1, *aux2; // tucker decomposition matrices
    double *res1, *res2; // restaurant lists
    size_t ncols, nrows, res1Size; // number of columns of sample
    const mwSize *coreDims, *aux1Dims, *aux2Dims;
    
    /* Check number of inputs and outputs */
    if(nrhs != 4) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                          "Four inputs required.");
    }
    if(nlhs != 1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs",
                          "One output required.");
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
    res1Size = mxGetN(mxGetCell(prhs[3],0));
    res2 = mxGetPr(mxGetCell(prhs[3],1));
    
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)nrows,(mwSize)ncols,mxREAL);

    /* get a pointer to the real data in the output matrix */
    sOut = mxGetPr(plhs[0]);
    
    /* call the computational routine */
    drawZsc(sIn,sOut,ncols,nrows,core,aux1,aux2,res1,res2,res1Size,coreDims,
            aux1Dims,aux2Dims);
}
