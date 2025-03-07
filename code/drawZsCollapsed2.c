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

#include "drawZsCollapsed.h"

void drawZs(double *sampIn, double *sampOut, double *p, size_t sampCols,
        size_t sampRows, double *phi, const mxArray *psi,
        double *pth, double *l, const mwSize *phiDims, int pri,
        double *cut, int topic, double *weights, double *var)
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
        alpha[0]= topic == 0 ? m/l[0]/l[1] : m/l[0];
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
    
    // get size
    modes -= 1;
    int size=1; int ma = 0;
    for(i=0; i<modes; i++){
        size *= floor(l[i]);
        ma = fmax(ma, floor(l[i]));
    }

    for(j=0; j<sampRows; j++){
        drawZ(j,sampIn,sampOut,p,sampCols,sampRows,phi,psi,psiSum,pth,l,
                phiDims,alpha,cut,topic,weights,modes,size,ma,var);  
    }
}

#if 0
void drawZsPar (double *sampIn, double *sampOut, double *p, size_t sampCols,
        size_t sampRows, double *phi, const mxArray *psi,
        double *pth, double *l, const mwSize *phiDims, int pri,
        double *cut, int topic, double *weights, double *var)
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
        alpha[0]= topic == 0 ? m/l[0]/l[1] : m/l[0];
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
    
    #pragma omp parallel private(j)
    {
        srand48(time(NULL)+omp_get_thread_num()); // randomize seed

        #pragma omp for
        for(j=0; j<sampRows; j++){
            drawZ(j,sampIn,sampOut,p,sampCols,sampRows,phi,psi,psiSum,pth,
                    l,phiDims,alpha,cut,topic,weights,modes,size,ma,var); 
        }
    }
}
#endif

void drawZ(int j, double *sampIn, double *sampOut, double *p,
        size_t sampCols, size_t sampRows, double *phi, const mxArray *psi,
        double *psis, double *pth, double *l, const mwSize *phiDims,
        double *a, double *cut, int topic, double *weights, int modes,
        int size, int ma, double *var)
{
    int x = sampIn[0*sampRows+j]-1; //get evidence variable
    int y; //response variable
    int zo; // get old topic
    
    // initialize sampOut
    long long int i; int k;
    for(i=0; i<sampCols; i++){
        sampOut[i*sampRows+j] = sampIn[i*sampRows+j];
    }
    
    // check dims
    for(k=0; k<modes; k++){
        if(phiDims[k+1]!=l[k] && phiDims[k+1]!=0 && round(l[k])!=1){
            mexPrintf("phiDims[%d] = %d\n", k+1, phiDims[k+1]);
            mexPrintf("l[%d] = %f\n", k, l[k]);
            mexErrMsgIdAndTxt("MyProg:dims:mismatch",
                "Dims of phi don't match number of levels.");
        }
    }
    
    double pdf1[modes*ma]; int s[modes*ma];
    double *psik; const mwSize *psikDims;
    int ind, ip, lsum, psiSum;
    lsum=0; psiSum=0;
    for(k=0; k<modes; k++){
        zo = sampIn[(k+1+modes)*sampRows+j]-1;
        y = sampIn[(k+1)*sampRows+j]-1;
        psik = mxGetPr(mxGetCell(psi,k));
        psikDims = mxGetDimensions(mxGetCell(psi,k));
        for(ind=0; ind<l[k]; ind++){
            ip = pth[x+(ind+lsum)*phiDims[0]]-1;
            s[ma*k+ind] = ((ip==zo) ? 1 : 0);
            pdf1[ma*k+ind] = log(psik[y+ip*psikDims[0]]+a[k+1]-s[ma*k+ind]);
            pdf1[ma*k+ind] -= log(psis[ip+psiSum]-s[ma*k+ind]);
            pdf1[ma*k+ind] *= weights[k];
            if(pdf1[ma*k+ind] != pdf1[ma*k+ind]) {
                mexPrintf("k = %d\n", k); mexPrintf("ind = %d\n", ind);
                mexPrintf("x = %d\n", x); mexPrintf("lsum = %d\n", lsum);
                mexPrintf("ip = %d\n", ip); mexPrintf("j = %d\n", j);
                mexPrintf("zo = %d\n", zo); mexPrintf("y = %d\n", y);
                mexPrintf("modes = %d\n", modes);
                mexPrintf("phi = %f\n", phi[x+i*phiDims[0]]);
                mexPrintf("a = %f\n", a[k+1]);
                mexPrintf("s = %d\n", s[ma*k+ind]);
                mexPrintf("psik = %f\n", psik[y+ip*psikDims[0]]);
                mexPrintf("psis = %f\n", psis[ip+psiSum]);
                mexPrintf("pdf1 = %f\n", pdf1[ma*k+ind]);
                mexErrMsgIdAndTxt("MyProg:sum:NaN", "Sum NaN.");
            }
        }
        var[k*sampRows+j]=variance(&pdf1[ma*k],l[k]);
        psiSum += psikDims[1];
        lsum += l[k];
    }
    
    double pdf[size]; double pdf2; double sum = 0; int st;
    for(i=0; i<size; i++){
        if (topic == 0 || indices(i,0,&phiDims[1]) == indices(i,1,&phiDims[1])){
            pdf2=0; st=1;
            for(k=0; k<modes; k++){
                ind = indices(i,k,&phiDims[1]);
                st *= s[ma*k+ind];
                pdf2 += pdf1[ma*k+ind];
            }
            pdf2 += log(phi[x+i*phiDims[0]]+a[0]-st);
            //mexPrintf("phi = %f\n", phi[x+i*phiDims[0]]);
            //mexPrintf("a = %f\n", a[0]); mexPrintf("st = %d\n", st);
            if(pdf2 != pdf2) {
                mexPrintf("j = %d\n", j);
                mexPrintf("x = %d\n", x); mexPrintf("i = %d\n", i);
                mexPrintf("phi = %f\n", phi[x+i*phiDims[0]]);
                mexPrintf("a = %f\n", a[0]); mexPrintf("st = %d\n", st);
                mexPrintf("pdf2 = %f\n", pdf2); mexPrintf("modes = %d\n", modes);
                mexErrMsgIdAndTxt("MyProg:sum:NaN", "Sum NaN.");
            }
            pdf[i] = exp(pdf2);
            pdf[i] = pdf[i] > *cut ? pdf[i] : 0.0;
            sum = sum + pdf[i];
            //mexPrintf("pdf[%d] = %f\n", i, pdf[i]);
            if(sum >= DBL_MAX) {
                mexPrintf("pdf2 = %f\n", pdf2);
                mexPrintf("pdf[%d] = %f\n", i, pdf[i]);
                mexErrMsgIdAndTxt("MyProg:sum:overflow", "Sum overflow.");
            }
            if(sum != sum) {
                mexPrintf("pdf2 = %f\n", pdf2);
                mexPrintf("pdf[%d] = %f\n", i, pdf[i]);
                mexErrMsgIdAndTxt("MyProg:sum:NaN", "Sum NaN.");
            }
        } else {
            pdf[i] = 0.0;
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
        t *= dims[i] == 0 ? 1 : dims[i];
    }
    o = x % t;
    t /= dims[m] == 0 ? 1 : dims[m];
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

double variance(double *pdf, int size){
    double pdfS=0; int i;
    double pdf2[size];
    for(i=0; i<size; i++){
        pdf2[i]=exp(pdf[i]);
        pdfS+=pdf2[i];
    }
    
    normalize(pdf2, pdfS, size);
    
    double var=0;
    for(i=0; i<size; i++){
        var+=pdf2[i]*(1-pdf2[i]);
    }
    var /= size;
    
    return var;
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
    char *topic; // Cartesian or Level
    double *weights; // exponent to weight modes
    double *var; // variance
    
    /* Check number of inputs and outputs */
    if(nrhs != 6) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                          "Seven inputs required.");
    }
    if(nlhs != 3) {
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
    prior = floor(*mxGetPr(mxGetField(prhs[5], 0, "pType")));
    cutoff = mxGetPr(mxGetField(prhs[5], 0, "cutoff"));
    topic = mxArrayToString(mxGetField(prhs[5], 0, "topicType"));
    weights = mxGetPr(mxGetField(prhs[5], 0, "weights"));
    
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)nrows,(mwSize)ncols,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(nrows,1,mxREAL);
    plhs[2] = mxCreateDoubleMatrix((mwSize)nrows,(mwSize)(ncols-1)/2,mxREAL);

    /* get a pointer to the real data in the output matrix */
    sOut = mxGetPr(plhs[0]);
    prob = mxGetPr(plhs[1]);
    var = mxGetPr(plhs[2]);
    
    int tb = strcmp(topic,"Cartesian");
    
    /* call the computational routine */
    drawZs(sIn,sOut,prob,ncols,nrows,core,aux,path,L,coreDims,prior,
            cutoff,tb,weights,var);
}
