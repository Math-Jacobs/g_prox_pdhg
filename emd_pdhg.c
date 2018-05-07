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


double *create_inverse_laplace_kernel(int n1, int n2){
    double *kernel=calloc(n1*n2,sizeof(double));
    for(int i=0;i<n2;i++){
        for(int j=0;j<n1;j++){
            double x=M_PI*j/(n1*1.0);
            double y=M_PI*i/(n2*1.0);
            
            double negativeLaplacian=2*n1*n1*(1-cos(x))+2*n2*n2*(1-cos(y));
            if(i>0||j>0){
                kernel[i*n1+j]=-1/(4*n1*n2*negativeLaplacian);
                
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
    fftps.kernel=create_inverse_laplace_kernel(n1,n2);
    
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



void invert_laplacian(poisson_solver fftps, double *phi, int n1, int n2){
    
    memcpy(fftps.workspace,phi,n1*n2*sizeof(double));
    
    fftw_execute(fftps.dctIn);
    
    for(int i=0;i<n1*n2;i++){
        fftps.workspace[i]*=fftps.kernel[i];
    }
    
    fftw_execute(fftps.dctOut);
    
    memcpy(phi,fftps.workspace,n1*n2*sizeof(double));
    
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

double update_p(double *px, double *py, double *ux, double *uy, double *uxold, double *uyold, double *psix, double *psiy,  double tau, double theta, int n1, int n2){
    double sigma=1/tau;
    double change=0;
    int pcount=n1*n2;
    for(int i=0;i<pcount;i++){
        double pxold=px[i];
        double pyold=py[i];
       
        px[i]+=sigma*(psix[i]+ux[i]+theta*(ux[i]-uxold[i]));
        py[i]+=sigma*(psiy[i]+uy[i]+theta*(uy[i]-uyold[i]));
        
        double norm=sqrt(px[i]*px[i]+py[i]*py[i]);
        
        double factor=fmax(norm,1);
        
        px[i]/=factor;
        py[i]/=factor;
        
        
        change+=pow(px[i]-pxold,2)+pow(py[i]-pyold,2);
        
        
    }
    change/=pcount;
    change=sqrt(change);
    return change;
}

void leray_projection(poisson_solver fftps, double *ux, double *uy, double *divergence, int n1, int n2){
    
    int pcount=n1*n2;
    
    double *tempx=calloc(pcount,sizeof(double));
    double *tempy=calloc(pcount,sizeof(double));
    
   
    memset(fftps.workspace,0,pcount*sizeof(double));
    
    calc_divergence(ux, uy, fftps.workspace, n1, n2);
    
    fftw_execute(fftps.dctIn);
    
    for(int i=0;i<pcount;i++){
        fftps.workspace[i]*=fftps.kernel[i];
    }
    
    fftw_execute(fftps.dctOut);
    
 
    calc_gradient(tempx, tempy, fftps.workspace , n1, n2);
    
    for(int i=0;i<pcount; i++){
        ux[i]-=tempx[i];
        uy[i]-=tempy[i];
        
    }
    
    free(tempx);
    free(tempy);
    
}




void update_u(poisson_solver fftps, double *ux, double *uy, double *px, double *py, double *divergence, double tau, int n1, int n2){
    int pcount=n1*n2;
    for(int i=0;i<pcount;i++){
        ux[i]-=tau*px[i];
        uy[i]-=tau*py[i];
    }
    leray_projection(fftps, ux, uy, divergence, n1, n2);
    
    
    
}




void check_divergence(double *density, double *divergence, double *psix, double *psiy, int n1, int n2){
    
    calc_divergence(psix, psiy, divergence, n1, n2);
    double sum=0;
    int pcount=n1*n2;
    for(int i=0;i<pcount;i++){
        sum+=pow(density[i]-divergence[i],2);
    }
    sum/=pcount;
    sum=sqrt(sum);
    
    printf("%e\n",sum);
    
    
}


void get_grad_psi(poisson_solver fftps, double *density, double *psix, double *psiy, int n1, int n2){
    
    int pcount=n1*n2;
    
    double *potential=calloc(pcount,sizeof(double));
    
    memcpy(potential, density,pcount*sizeof(double));
    
    invert_laplacian(fftps, potential, n1, n2);
    
    calc_gradient(psix, psiy, potential, n1, n2);
    
    check_divergence(density, potential, psix, psiy, n1, n2);
    
    free(potential);
}


double calc_primal(double *ux, double *uy, double *psix, double *psiy, int n1, int n2){
    
    int pcount=n1*n2;
    double value=0;
    for(int i=0;i<pcount;i++){
        double norm=sqrt(pow(ux[i]+psix[i],2)+pow(uy[i]+psiy[i],2));
        value+=norm;
    }
    value/=pcount;
    return value;
    
}

double calc_dual(double *px, double *py, double *psix, double *psiy,  int n1, int n2){
    
    int pcount=n1*n2;
    
    
    double value=0;
    for(int i=0;i<pcount;i++){
        value+=px[i]*psix[i]+py[i]*psiy[i];
        
    }
    value/=pcount;
    return value;
    
}







void emd(double *density, double tau, double error, int maxIters, int n1, int n2){
    int pcount=n1*n2;
    poisson_solver fftps=create_poisson_solver_workspace(n1, n2);
   
    
    double *ux=calloc(pcount,sizeof(double));
    double *uy=calloc(pcount,sizeof(double));
    
    double *uxold=calloc(pcount,sizeof(double));
    double *uyold=calloc(pcount,sizeof(double));

    double *px=calloc(pcount,sizeof(double));
    double *py=calloc(pcount,sizeof(double));
    
    double *psix=calloc(pcount,sizeof(double));
    double *psiy=calloc(pcount,sizeof(double));
    
    double *divergence=calloc(pcount,sizeof(double));
    
    get_grad_psi(fftps, density, psix, psiy, n1, n2);
    
    double theta=1;
    int smaller=0;
    
    for(int i=0;i<maxIters;i++){
        update_p(px, py, ux, uy, uxold, uyold, psix, psiy,  tau, theta, n1, n2);
        
        memcpy(uxold,ux,pcount*sizeof(double));
        memcpy(uyold,uy,pcount*sizeof(double));
        
        update_u(fftps, ux, uy, px, py, divergence, tau, n1, n2);
        
        
        double primal=calc_primal(ux, uy, psix, psiy, n1, n2);
        double dual=calc_dual(px, py, psix, psiy, n1, n2);
        
       
        if(fabs(primal-dual)<error){
            printf("broken at iteration %d\n",i);
            break;
        }
       
        
    }
    
    
    
    destroy_poisson_solver(fftps);
   
    
    free(ux);
    free(uy);
    free(uxold);
    free(uyold);
   
    free(psix);
    free(psiy);
    free(px);
    free(py);
    free(divergence);
   
    
    
    
    
}
















