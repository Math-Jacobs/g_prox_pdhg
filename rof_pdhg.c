
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <stdio.h>
#include <fftw3.h>


typedef struct{
    fftw_plan dctIn;
    fftw_plan dctOut;
    double *kernel;
    double *workspace;
}poisson_solver;


double *create_negative_laplace_kernel(int n1, int n2){
    double *kernel=calloc(n1*n2,sizeof(double));
    for(int i=0;i<n2;i++){
        for(int j=0;j<n1;j++){
            double x=M_PI*j/(n1*1.0);
            double y=M_PI*i/(n2*1.0);
            
            double negativeLaplacian=2*n1*n1*(1-cos(x))+2*n2*n2*(1-cos(y));
            if(i>0||j>0){
                kernel[i*n1+j]=negativeLaplacian;
                
            }
            
        }
    }
    return kernel;
}


poisson_solver create_poisson_solver_workspace(int n1, int n2){
    clock_t b,e;
    b=clock();
    poisson_solver fftps;
    fftps.workspace=calloc(n1*n2,sizeof(double));
    fftps.kernel=create_negative_laplace_kernel(n1,n2);
    
    fftps.dctIn=fftw_plan_r2r_2d(n2, n1, fftps.workspace, fftps.workspace,
                                 FFTW_REDFT10, FFTW_REDFT10,
                                 FFTW_MEASURE);
    fftps.dctOut=fftw_plan_r2r_2d(n2, n1, fftps.workspace, fftps.workspace,
                                  FFTW_REDFT01, FFTW_REDFT01,
                                  FFTW_MEASURE);
    
    e=clock();
    
    printf("fft setup time: %f\n", (e-b)/(CLOCKS_PER_SEC*1.0));
    
    return fftps;
}



void destroy_poisson_solver(poisson_solver fftps){
    free(fftps.kernel);
    free(fftps.workspace);
    fftw_destroy_plan(fftps.dctIn);
    fftw_destroy_plan(fftps.dctOut);
}


double calc_divergence(double *ux, double *uy, double *divergence, int n1, int n2){
    double divTot=0;
    for(int i=0;i<n2-1;i++){
        for(int j=0;j<n1-1;j++){
            
            int xp=(j+1);
            int yp=(i+1);
            
            divergence[i*n1+j]=n1*(ux[i*n1+xp]-ux[i*n1+j])+n2*(uy[yp*n1+j]-uy[i*n1+j]);
            divTot+=pow(divergence[i*n1+j],2);
        }
        divergence[i*n1+n1-1]=-n1*ux[i*n1+n1-1]+n2*(uy[(i+1)*n1+n1-1]-uy[i*n1+n1-1]);
        divTot+=pow(divergence[i*n1+n1-1],2);
    }
    for(int j=0;j<n1-1;j++){
        divergence[(n2-1)*n1+j]=n1*(ux[(n2-1)*n1+j+1]-ux[(n2-1)*n1+j])-n2*uy[(n2-1)*n1+j];
        divTot+=pow(divergence[(n2-1)*n1+j],2);
        
    }
    divergence[n2*n1-1]=-(n1*ux[n1*n2-1]+n2*uy[n1*n2-1]);
    divTot+=pow(divergence[n2*n1-1],2);
    
    divTot/=n1*n2;
    divTot=sqrt(divTot);
    return divTot;
}

void calc_gradient(double *ux, double *uy, double *potential, int n1, int n2){
    
    ux[0]=0;
    memset(uy,0,n1*sizeof(double));
    for(int j=1;j<n1;j++){
        ux[j]=n1*(potential[j]-potential[j-1]);
    }
    
    
    for(int i=1;i<n2;i++){
        ux[n1*i]=0;
        uy[n1*i]=n2*(potential[i*n1]-potential[(i-1)*n1]);
        for(int j=1;j<n1;j++){
            int xm=j-1;
            int ym=i-1;
            ux[i*n1+j]=n1*(potential[i*n1+j]-potential[i*n1+xm]);
            uy[i*n1+j]=n2*(potential[i*n1+j]-potential[ym*n1+j]);
        }
    }
}



double update_u(poisson_solver fftps, double *u, double *px, double *py, double *f, double lambda, double tau, int n1, int n2){
    
    int pcount=n1*n2;
    
    double *tempx=calloc(pcount,sizeof(double));
    double *tempy=calloc(pcount,sizeof(double));
    
    calc_gradient(tempx, tempy, u, n1, n2);
    
    
    for(int i=0;i<pcount;i++){
        tempx[i]-=tau*px[i];
        tempy[i]-=tau*py[i];
    }
    
    calc_divergence(tempx, tempy, fftps.workspace, n1, n2);
    
    for(int i=0;i<pcount;i++){
        fftps.workspace[i]-=tau*lambda*f[i];
    }
    
    fftw_execute(fftps.dctIn);
    
    for(int i=0;i<pcount;i++){
        
        fftps.workspace[i]/=4*pcount*(fftps.kernel[i]+tau*lambda);
        
    }
    
    fftw_execute(fftps.dctOut);
    
    double change=0;
    
    for(int i=0;i<pcount;i++){
        double old=u[i];
        u[i]=-fftps.workspace[i];
        change+=pow(u[i]-old,2);
    }
    
    free(tempx);
    free(tempy);
    
    change/=pcount;
    change=sqrt(change);
    
    return change;
}



void update_p(double *px, double *py, double *u, double *uold, double sigma, double theta, int n1, int n2){
    int pcount=n1*n2;
    
   
    for(int i=0; i<n2;i++){
        for(int j=1;j<n1;j++){
            
            double ux=n1*(u[i*n1+j]-u[i*n1+j-1]);
            double uxold=n1*(uold[i*n1+j]-uold[i*n1+j-1]);
            
            px[i*n1+j]+=sigma*(ux+theta*(ux-uxold));
        }
    }
    
    for(int i=1; i<n2;i++){
        for(int j=0;j<n1;j++){
            
            double uy=n2*(u[i*n1+j]-u[(i-1)*n1+j]);
            double uyold=n2*(uold[i*n1+j]-uold[(i-1)*n1+j]);
            
            py[i*n1+j]+=sigma*(uy+theta*(uy-uyold));
        }
    }
   
    for(int i=0;i<pcount;i++){
        double norm=sqrt(px[i]*px[i]+py[i]*py[i]);
        double factor=fmax(norm,1);
        px[i]/=factor;
        py[i]/=factor;
        
    }
}


double calc_primal(double *u, double *f, double lambda, int n1, int n2){
    
    double primal=0;
    
    int pcount=n1*n2;
    
    double *tempx=calloc(pcount,sizeof(double));
    double *tempy=calloc(pcount,sizeof(double));

    calc_gradient(tempx, tempy, u, n1, n2);
    
    for(int i=0;i<pcount;i++){
        
        double norm=sqrt(tempx[i]*tempx[i]+tempy[i]*tempy[i]);
        primal+=norm+lambda*pow(u[i]-f[i],2)/2;
        
    }
    free(tempx);
    free(tempy);
    
    primal/=pcount;
    
    return primal;
}

double calc_dual(double *px, double *py, double *f, double lambda, int n1, int n2){
    
    int pcount=n1*n2;
    double dual=0;
    double *divergence=calloc(pcount,sizeof(double));
    
    calc_divergence(px, py, divergence, n1, n2);
    
    for(int i=0;i<pcount;i++){
        
        dual+=-f[i]*divergence[i]-.5*pow(divergence[i],2)/lambda;
        
    }
    
    free(divergence);
    
    dual/=pcount;
    
    return dual;
}






void rof(double *u, double *f, double lambda, double tau, double error, int maxIters, int n1, int n2){
    int pcount=n1*n2;
    poisson_solver fftps=create_poisson_solver_workspace(n1, n2);
    
    double *px=calloc(pcount,sizeof(double));
    double *py=calloc(pcount,sizeof(double));
   
    double *uold=calloc(pcount,sizeof(double));
    
    
    
    double theta=1;

    double sigma=1/tau;
    
    for(int i=0;i<maxIters;i++){
        
        memcpy(uold,u,pcount*sizeof(double));
        
        update_u(fftps, u, px, py, f, lambda, tau, n1, n2);
        
        update_p(px, py, u, uold, sigma, theta, n1, n2);
        
        
        
        double primal, dual;
        
        primal=calc_primal(u, f, lambda, n1, n2);
        
        dual= calc_dual(px, py, f, lambda, n1, n2);
        
        if(i%10==1){
            printf("%f %f %e \n",primal, dual, primal-1.18316);
        }
        
        if(fabs(primal-1.18316)<error){
            printf("broken at iteration %d\n",i);
            break;
        }
        
        
    }
    
    
    
    destroy_poisson_solver(fftps);
    free(px);
    free(py);
    free(uold);
    
    
    
}











