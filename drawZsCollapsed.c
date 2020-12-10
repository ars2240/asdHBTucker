/*
  drawZsCollapsed.c

  draws latent topic z's
  samp = sample matrix
  cphi = tucker decomposition tensor core tensor
  cpsi = tucker decomposition matrices
  r = restaurant lists / paths per each x
  prior = 1 or 1/L
 
 The calling syntax is:
    [samp,p] = drawZsCollapsedPar(samp,cphi,cpsi,path,L,prior)

  Created by Adam Sandler.
*/

#include "mex.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// forward declarations
void drawZs(double *sampIn, double *sampOut, double *p, size_t sampCols,
        size_t sampRows, double *phi, const mxArray *psi,
        double *pth, double *l, const mwSize *phiDims, int pri, double *cut);
void drawZ(int j, double *sampIn, double *sampOut, double *p,
        size_t sampCols, size_t sampRows, double *phi, const mxArray *psi,
        double *psis, double *pth, double *l, const mwSize *phiDims,
        double *a, double *cut);
int indices(long long int x, int m, const mwSize *dims);
void normalize(double *pdf, double sum, int size);
long long int multi(double *pdf, int size);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
        const mxArray *prhs[]);

void drawZs(double *sampIn, double *sampOut, double *p, size_t sampCols,
        size_t sampRows, double *phi, const mxArray *psi,
        double *pth, double *l, const mwSize *phiDims, int pri, double *cut)
{
    int j, i, k;
    
    // setup alphas
    int modes = (sampCols+1)/2;
    double alpha[modes]; const mwSize *psikDims;
    if(pri==1){
        for(i=0; i<modes; i++){
            alpha[i]=1.0;
        }
    } else{
        double m = pri==2 ? 2.0 : 1.0;
        alpha[0]=m/l[0]/l[1];
        for(i=1; i<modes; i++){
            psikDims = mxGetDimensions(mxGetCell(psi,i-1));
            alpha[i]=m/psikDims[0];
        }
    }
    int psiDimSum=0; int psiDimPos=0;
    for(i=1; i<modes; i++){
        psikDims = mxGetDimensions(mxGetCell(psi,i-1));
        psiDimSum+=psikDims[1];
    }
    double psiSum[psiDimSum]; double *psik;
    for(i=1; i<modes; i++){
        psik = mxGetPr(mxGetCell(psi,i-1));
        psikDims = mxGetDimensions(mxGetCell(psi,i-1));
    	for(j=0; j<psikDims[1]; j++){
            psiSum[j+psiDimPos]=alpha[i]*psikDims[0];
            for(k=0; k<psikDims[0]; k++){
                psiSum[j+psiDimPos]+=psik[k+j*psikDims[0]];
            }
        }
        psiDimPos+=psikDims[1];
    }
    srand48(time(NULL)); // randomize seed

    for(j=0; j<sampRows; j++){
        drawZ(j,sampIn,sampOut,p,sampCols,sampRows,phi,psi,psiSum,pth,l,
                phiDims,alpha,cut);  
    }
}

void drawZ(int j, double *sampIn, double *sampOut, double *p,
        size_t sampCols, size_t sampRows, double *phi, const mxArray *psi,
        double *psis, double *pth, double *l, const mwSize *phiDims,
        double *a, double *cut)
{
    int x = sampIn[0*sampRows+j]-1; //get evidence variable
    int y; //response variable
    int zo; // get old topic
    
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
    double pdf[size]; double pdf1; double sum = 0;
    int ind, ip, s, st, lsum, psiSum;
    double *psik; const mwSize *psikDims;
    for(i=0; i<size; i++){
        pdf1=0; st=1; lsum=0; psiSum=0;
        for(k=0; k<modes; k++){
            ind = indices(i,k,&phiDims[1]);
            ip = pth[x+(ind+lsum)*phiDims[0]]-1;
            lsum += l[k];
            zo = sampIn[(1+modes+k)*sampRows+j]-1;
            y = sampIn[(k+1)*sampRows+j]-1;
            s = ((ip==zo) ? 1 : 0);
            st *= s;
            psik = mxGetPr(mxGetCell(psi,k));
            psikDims = mxGetDimensions(mxGetCell(psi,k));
            pdf1 += log(psik[y+ip*psikDims[0]]+a[k+1]-s);
            pdf1 -= log(psis[ip+psiSum]-s);
            if(pdf1 != pdf1) {
                mexPrintf("i = %d\n", i); mexPrintf("k = %d\n", k);
                mexPrintf("ind = %d\n", ind); mexPrintf("x = %d\n", x);
                mexPrintf("lsum = %d\n", lsum); mexPrintf("ip = %d\n", ip);
                mexPrintf("j = %d\n", j); mexPrintf("zo = %d\n", zo);
                mexPrintf("y = %d\n", y); mexPrintf("modes = %d\n", modes);
                mexPrintf("phi = %f\n", phi[x+i*phiDims[0]]);
                mexPrintf("a = %f\n", a[k+1]); mexPrintf("s = %d\n", s);
                mexPrintf("psik = %f\n", psik[y+ip*psikDims[0]]);
                mexPrintf("psis = %f\n", psis[ip+psiSum]);
                mexPrintf("pdf1 = %f\n", pdf1);
                mexErrMsgIdAndTxt("MyProg:sum:NaN", "Sum NaN.");
            }
            psiSum += psikDims[1];
        }
        pdf1 += log(phi[x+i*phiDims[0]]+a[0]-st);
        //mexPrintf("phi = %f\n", phi[x+i*phiDims[0]]);
        //mexPrintf("a = %f\n", a[0]); mexPrintf("st = %d\n", st);
        if(pdf1 != pdf1) {
            mexPrintf("j = %d\n", j);
            mexPrintf("x = %d\n", x); mexPrintf("i = %d\n", i);
            mexPrintf("phi = %f\n", phi[x+i*phiDims[0]]);
            mexPrintf("a = %f\n", a[0]); mexPrintf("st = %d\n", st);
            mexPrintf("pdf1 = %f\n", pdf1); mexPrintf("modes = %d\n", modes);
            pdf1=0; st=1; lsum=0; psiSum=0;
            for(k=0; k<modes; k++){
                ind = indices(i,k,&phiDims[1]);
                ip = pth[x+(ind+lsum)*phiDims[0]]-1;
                lsum += l[k];
                zo = sampIn[(1+modes+k)*sampRows+j]-1;
                y = sampIn[(k+1)*sampRows+j]-1;
                s = ((ip==zo) ? 1 : 0);
                psik = mxGetPr(mxGetCell(psi,k));
                psikDims = mxGetDimensions(mxGetCell(psi,k));
                mexPrintf("k = %d\n", k); mexPrintf("ind = %d\n", ind);
                mexPrintf("lsum = %d\n", lsum); mexPrintf("ip = %d\n", ip);
                mexPrintf("zo = %d\n", zo); mexPrintf("y = %d\n", y);
                mexPrintf("a = %f\n", a[k+1]); mexPrintf("s = %d\n", s);
                mexPrintf("psik = %f\n", psik[y+ip*psikDims[0]]);
                mexPrintf("psis = %f\n", psis[ip+psiSum]);
            }
            mexErrMsgIdAndTxt("MyProg:sum:NaN", "Sum NaN.");
        }
        pdf[i] = exp(pdf1);
        pdf[i] = pdf[i] > *cut ? pdf[i] : 0.0;
        sum = sum + pdf[i];
        //mexPrintf("pdf[%d] = %f\n", i, pdf[i]);
        if(sum >= DBL_MAX) {
            mexPrintf("pdf1 = %f\n", pdf1);
            mexPrintf("pdf[%d] = %f\n", i, pdf[i]);
            mexErrMsgIdAndTxt("MyProg:sum:overflow", "Sum overflow.");
        }
        if(sum != sum) {
            mexPrintf("pdf1 = %f\n", pdf1);
            mexPrintf("pdf[%d] = %f\n", i, pdf[i]);
            mexErrMsgIdAndTxt("MyProg:sum:NaN", "Sum NaN.");
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
        if(round(pth[x+(z1+lsum)*phiDims[0]]) ==0){
            mexPrintf("k = %d\n", k);
            mexPrintf("x = %d\n", x);
            mexPrintf("z = %d\n", z);
            mexPrintf("z1 = %d\n", z1);
            mexPrintf("ind = %d\n", x+(z1+lsum)*phiDims[0]);
            mexErrMsgIdAndTxt("MyProg:badVal:badTopic", "Bad Topic.");
        }
        lsum += l[k];
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
        if(pdf[i]/sum != pdf[i]/sum){
            mexPrintf("pdf[%d] = %f\n", i, pdf[i]);
            mexPrintf("sum = %f\n", sum);
            mexErrMsgIdAndTxt("MyProg:badSum:badSum", "Bad Sum.");
        }
        pdf[i] = pdf[i]/sum;
        //mexPrintf("pdf[%d] = %f\n", i, pdf[i]);
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
    
    if(found == 0){
        mexPrintf("cdf = %f\n", cdf[i-1]);
        mexPrintf("n = %f\n", n);
        mexErrMsgIdAndTxt("MyProg:badVal:badTopic", "Bad Topic.");
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
    int prior; // prior
    size_t ncols, nrows, res2Size; // number of columns of sample
    const mwSize *coreDims;
    double *cutoff; // zero out values < cutoff
    
    /* Check number of inputs and outputs */
    if(nrhs != 7) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                          "Seven inputs required.");
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
    prior = floor(*mxGetPr(prhs[5]));
    cutoff = mxGetPr(prhs[6]);
    
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)nrows,(mwSize)ncols,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(nrows,1,mxREAL);

    /* get a pointer to the real data in the output matrix */
    sOut = mxGetPr(plhs[0]);
    prob = mxGetPr(plhs[1]);
    
    /* call the computational routine */
    drawZs(sIn,sOut,prob,ncols,nrows,core,aux,path,L,coreDims,prior,cutoff);
}
