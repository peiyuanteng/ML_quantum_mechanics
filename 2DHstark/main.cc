

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <complex>
#include <fstream>
#include <random>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <stdlib.h>
#include <time.h>
using namespace Eigen;
//Simple Monte Carlo sampling of a spin
//Wave-Function

template<typename _Matrix_Type_>
_Matrix_Type_ pseudoInverse(const _Matrix_Type_ &a, double epsilon = std::numeric_limits<double>::epsilon())
{
    JacobiSVD< _Matrix_Type_ > svd(a , ComputeThinU | ComputeThinV);
    double tolerance = epsilon * std::max(a.cols(), a.rows()) *svd.singularValues().array().abs()(0);
    return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
}


class Wavefunctionradial{
    
private:
    //Neural-network
    std::vector<std::complex<double> > a;
    std::vector<std::complex<double> > b;
    std::vector<std::vector<std::complex<double> >> c;

    
    //Number of neurons
    int n;
    int m;
    

    
public:
    
    Wavefunctionradial(int n_,int m_):n(n_),m(m_){
        a.resize(n);
        b.resize(n);
        c.resize(n,std::vector<std::complex<double> > (m));
        
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> disb(0, 1);
        std::uniform_real_distribution<> dis(0, 1);
        

        

        //std::cout<<c<<std::endl;
        for(int i=0;i<n;i++){
            a[i]=dis(gen);
            b[i]=1.0;
            for(int j=0;j<m;j++){
                c[i][j]=dis(gen);}
        }
        
    }
    
    //sigmoid
    std::complex<double> radial(std::vector<double> & r,int i)
    {
        std::complex<double> result(0,0);
        for(int j=0;j<m;j++){
            result+=(r[j]-c[i][j])*(r[j]-c[i][j]);
        }
    
        return  exp(- std::abs(b[i])* result);
    }

    std::complex<double> radialapp(std::vector<double> & r,int i)
    {
        std::complex<double> result(0,0);
        for(int j=0;j<m;j++){
            result+=2.0*a[i]*std::abs(b[i])*radial(r,i)*(2.0*std::abs(b[i]) * (r[j]-c[i][j])* (r[j]-c[i][j])-1.0);
        }
        
        return result;
    }

    //wavefunction
    std::complex<double> psi(std::vector<double> & r)
    {
        std::complex<double> result(0,0);
        for(int i=0;i<n;i++){
            result+=a[i]*radial(r,i);
        }
        return result;
    }
    

    
    std::complex<double> psipp(std::vector<double> & r)
    {
        std::complex<double> result(0,0);
        for(int i=0;i<n;i++){
            result+=radialapp(r,i);
        }
        return result;
    }
    
    
    //update the parameters of the wave-function
    void UpdateParameters(const std::vector<std::complex<double> > & da, const std::vector<std::complex<double> > & db,const std::vector<std::vector<std::complex<double> >>  & dc){
        

        for(int i=0;i<n;i++){
            a[i]+=da[i];

            b[i]+=db[i];
            for(int j=0;j<m;j++){
                c[i][j]+=dc[i][j];
        }

        }


    }
    
    void calvp(std::vector<double> & r, std::vector<std::complex<double> > & vp)
    {
        vp.resize(2*n+n*m);
        std::complex<double> p0=psi(r);

        
        //a
        for(int i=0;i<n;i++){
            vp[i]=radial(r,i)/p0;
        }
        
        //b
        for(int i=0;i<n;i++){
            std::complex<double> result(0,0);
            for(int j=0;j<m;j++){
                result+=(r[j]-c[i][j])*(r[j]-c[i][j]);
            }
            vp[i+n]=-a[i]*radial(r,i)*result*b[i]/p0/std::abs(b[i]);
        }
        
        //c
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                vp[j+i*m+n*2]=2.0*a[i]*radial(r,i)*(r[j]-c[i][j])*std::abs(b[i])/p0;
            }

        }
        
    }
    
    //get parameters
    std::vector<std::complex<double> >  geta(){ return a;}
    
    std::vector<std::complex<double> > getb(){ return b;}
    
    std::vector<std::vector<std::complex<double> >> getc(){ return c;}
    
    
    int getsizea(){ return a.size();}
    
    int getsizeb(){ return b.size();}

    int getsizec(){ return c.size();}
    
    int getn(){ return n;}
    int getm(){ return m;}

    
};



class Wavefunctionabs{
    
private:
    //Neural-network
    std::vector<std::complex<double> > a;
    std::vector<std::complex<double> > b;
    std::vector<std::vector<std::complex<double> >> c;
    
    
    //Number of n and wavefunction
    int n;
    int m;
    
    
    
public:
    
    Wavefunctionabs(int n_,int m_):n(n_),m(m_){
        a.resize(n);
        b.resize(n);
        c.resize(n,std::vector<std::complex<double> > (m));
        
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> disb(0, 1);
        std::uniform_real_distribution<> dis(0, 1);
        
        
        //std::cout<<c<<std::endl;
        for(int i=0;i<n;i++){
            a[i]=dis(gen);
            b[i]=1.0;
            for(int j=0;j<m;j++){
                c[i][j]=dis(gen);}
        }
        
    }
    
    //sigmoid
    std::complex<double> radial(std::vector<double> & r,int i)
    {
        std::complex<double> result(0,0);
        for(int j=0;j<m;j++){
            result+=(r[j]-c[i][j])*(r[j]-c[i][j]);
        }
        
        result=exp(- std::abs(b[i])* std::sqrt(result));
        
        return result;
    }
    
    std::complex<double> radialp(std::vector<double> & r,int i)
    {
        std::complex<double> result(0,0);
        for(int j=0;j<m;j++)
        {
            result+=-(r[j]-c[i][j])/std::abs(r[j]-c[i][j])/std::abs(r[j]);
            
        }
        
        
        
        result*=a[i]*radial(r,i)*std::abs(b[i])*2.0;
        
        return result;
    }
    
    
    std::complex<double> radialpp(std::vector<double> & r,int i)
    {
        std::complex<double> result(0,0);
        
        for(int j=0;j<m;j++){
            result+=a[i]*radial(r,i)*std::abs(b[i])*std::abs(b[i]);
        }
        
        return result;
    }
    
    //wavefunction
    std::complex<double> psi(std::vector<double> & r)
    {
        std::complex<double> result(0,0);
        for(int i=0;i<n;i++){
            result+=a[i]*radial(r,i);
        }
        return result;
    }
    
    
    std::complex<double> psip(std::vector<double> & r)
    {
        std::complex<double> result(0,0);
        for(int i=0;i<n;i++){
            result+=radialp(r,i);
        }
        return result;
    }
    
    
    
    std::complex<double> psipp(std::vector<double> & r)
    {
        std::complex<double> result(0,0);
        for(int i=0;i<n;i++){
            result+=radialpp(r,i);
        }
        return result;
    }
    
    
    //update the parameters of the wave-function
    void UpdateParameters(const std::vector<std::complex<double> > & da, const std::vector<std::complex<double> > & db,const std::vector<std::vector<std::complex<double> >>  & dc){
        
        
        for(int i=0;i<n;i++){
            a[i]+=da[i];
            
            b[i]+=db[i];
            for(int j=0;j<m;j++){
                c[i][j]+=dc[i][j];
            }
            
        }
        
        
    }
    
    void calvp(std::vector<double> & r, std::vector<std::complex<double> > & vp)
    {
        vp.resize(2*n+n*m);
        std::complex<double> p0=psi(r);
        
        
        //a
        for(int i=0;i<n;i++){
            vp[i]=radial(r,i)/p0;
        }
        
        //b
        for(int i=0;i<n;i++){
            std::complex<double> result(0,0);
            for(int j=0;j<m;j++){
                result+=std::abs(r[j]-c[i][j]);
            }
            vp[i+n]=-a[i]*radial(r,i)*result*b[i]/p0/std::abs(b[i]);
        }
        
        //c
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                vp[j+i*m+n*2]=a[i]*radial(r,i)*(r[j]-c[i][j])*std::abs(b[i])/std::abs(r[j]-c[i][j])/p0;
            }
            
        }
        
    }
    
    //get parameters
    std::vector<std::complex<double> >  geta(){ return a;}
    
    std::vector<std::complex<double> > getb(){ return b;}
    
    std::vector<std::vector<std::complex<double> >> getc(){ return c;}
    
    
    int getsizea(){ return a.size();}
    
    int getsizeb(){ return b.size();}
    
    int getsizec(){ return c.size();}
    
    int getn(){ return n;}
    int getm(){ return m;}
    
    
};




class HamiltonianInteract
{
private:
    double omega;
    double mass;
    double lambda;
public:
    HamiltonianInteract(double o, double m, double l):omega(o),mass(m),lambda(l){}

    std::complex<double> matrixelement(std::vector<double> & na)
    {
        

        
        std::complex<double> result(0,0);
        for(int k=0;k<na.size();k++)
        {
            result+=(na[k]+0.5)*omega;
        }
        return result;
    }
    
    
    
};



int main(int argc, char *argv[]){

    Wavefunctionradial Wave(10,2);
    std::ofstream myfile;
    myfile.open ("data.txt");
    
    double lambda1=4.0;
    double lambda2=2.0;
    double lambda=0.5;
    HamiltonianInteract H(1,1,lambda);
    
    //random number
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-1, 1);
    std::uniform_real_distribution<> sample(0, 1);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);
    std::uniform_int_distribution<int> disi(0, 1);
    srand(time(NULL));


    
    int s=Wave.getn();
    int sm=Wave.getm();
    int st=sm*s+2*s;

    int nmax=40;

    
    
    int step=200;
    
    
    for(int loopopt=0;loopopt<step;loopopt++)
    {

        //reconfiguration setup
        Matrix<std::complex<double>, Dynamic, Dynamic> opo;
        Matrix<std::complex<double>, Dynamic, 1> o;
        Matrix<std::complex<double>, Dynamic, 1> op;
        Matrix<std::complex<double>, Dynamic, 1> ep;

        std::vector<std::complex<double> > para;
        
        Matrix<std::complex<double>, Dynamic, Dynamic> S;
        Matrix<std::complex<double>, Dynamic, 1> F;
        
        para.resize(st);
        opo.resize(st,st);
        o.resize(st,1);
        op.resize(st,1);
        ep.resize(st,1);
        
        S.resize(st,st);
        F.resize(st,1);
        
        for(int i=0;i<st;i++)
        {   o(i,0)=0;
            op(i,0)=0;
            ep(i,0)=0;
            for(int j=0;j<st;j++)
            { opo(i,j)=0;}
        }
        
        //position
        
        int nsweeps=50000;
        int flip1=0;
        int flip2=0;
        int accept=0;
        std::vector<double> state(2);
        std::vector<double> staten(2);
        std::vector<double> statem(2);
        state[0]=rand()%nmax;
        state[1]=rand()%nmax;
        state=staten;

        for(int n=0;n<nsweeps*0.05;n+=1)
        {
            flip1=disi(gen);
            flip2=(disi(gen)-0.5)*2;
            staten[flip1]=state[flip1]+flip2;
            
            if (staten[0]<0) staten[0]=0;
            if (staten[0]>=nmax) staten[0]=nmax-1;
            if (staten[1]<0) staten[1]=0;
            if (staten[1]>=nmax) staten[1]=nmax-1;
            
            std::complex<double> w = Wave.psi(staten) / Wave.psi(state);
            
            if (sample(gen)< std::norm(w)) {
                
                state=staten; accept++;
            }
            
        }

        
  
        accept=0;
        
        
        
        std::complex<double> energy(0,0);

        
        for(int n=0;n<nsweeps;n+=1)
        {
            flip1=disi(gen);
            flip2=(disi(gen)-0.5)*2;
            staten[flip1]=state[flip1]+flip2;
            
            if (staten[0]<0) staten[0]=0;
            if (staten[0]>=nmax) staten[0]=nmax-1;
            if (staten[1]<0) staten[1]=0;
            if (staten[1]>=nmax) staten[1]=nmax-1;
            
            
            std::complex<double> w = Wave.psi(staten) / Wave.psi(state);
            //std::cout<<w<<std::endl;
            if (sample(gen)<std::norm(w)) {
                
                state[0]=staten[0];
                state[1]=staten[1];
                accept++;
            }
            
           // std::cout<<staten[0]<<" "<<staten[1]<<std::endl;
            std::complex<double> E(0,0);
            
            E=H.matrixelement(state);
            
            double ttt1=0.0;
            double ttt2=0.0;

            
            
             if ((state[0]<0.1)&&(state[1]<0.1))
             {   statem=state;
             statem[0]=state[0]+1; ttt1=std::sqrt((statem[0])/2.0);
             E+=-ttt1*lambda1*Wave.psi(statem).real()/Wave.psi(state).real();
             statem=state;
             statem[1]=state[1]+1; ttt2=std::sqrt((statem[1])/2.0);
             E+=-(double)ttt2*(double)lambda2*double(Wave.psi(statem).real())/(double)Wave.psi(state).real();
             
             }
             else if ((state[0]<0.1)&&(state[1]>=0.1))
             {   statem=state;
             statem[0]=state[0]+1; ttt1=std::sqrt((statem[0])/2.0);
             E+=-ttt1*lambda1*Wave.psi(statem).real()/Wave.psi(state).real();
             
             statem=state;
             
             statem[1]=state[1]+1; ttt2=std::sqrt((statem[1])/2.0);
             
             E+=-(double)ttt2*(double)lambda2*double(Wave.psi(statem).real())/(double)Wave.psi(state).real();
             statem[1]=state[1]-1; ttt2=std::sqrt((state[1])/2.0);
             E+=-(double)ttt2*(double)lambda2*double(Wave.psi(statem).real())/(double)Wave.psi(state).real();
             
             
             }
             
             else if ((state[0]>=0.1)&&(state[1]<0.1))
             {   statem=state;
             statem[0]=state[0]+1; ttt1=std::sqrt((statem[0])/2.0);
             
             E+=-ttt1*lambda1*Wave.psi(statem).real()/Wave.psi(state).real();
             statem[0]=state[0]-1; ttt1=std::sqrt((state[0])/2.0);
             
             E+=-ttt1*lambda1*Wave.psi(statem).real()/Wave.psi(state).real();
             
             statem=state;
             statem[1]=state[1]+1; ttt2=std::sqrt((statem[1])/2.0);
             E+=-(double)ttt2*(double)lambda2*double(Wave.psi(statem).real())/(double)Wave.psi(state).real();
             }
             else if ((state[0]>=0.1)&&(state[1]>=0.1))
             {   statem=state;
             statem[0]=state[0]+1; ttt1=std::sqrt((statem[0])/2.0);
             
             E+=-ttt1*lambda1*Wave.psi(statem).real()/Wave.psi(state).real();
             statem[0]=state[0]-1; ttt1=std::sqrt((state[0])/2.0);
             
             E+=-ttt1*lambda1*Wave.psi(statem).real()/Wave.psi(state).real();
             
             statem=state;
             
             statem[1]=state[1]+1; ttt2=std::sqrt((statem[1])/2.0);
             
             E+=-(double)ttt2*(double)lambda2*double(Wave.psi(statem).real())/(double)Wave.psi(state).real();
             statem[1]=state[1]-1; ttt2=std::sqrt((state[1])/2.0);
             
             E+=-(double)ttt2*(double)lambda2*double(Wave.psi(statem).real())/(double)Wave.psi(state).real();
             
             }
             
            
            
            /*
            if ((state[0]<0.1)&&(state[1]<0.1))
            {   statem=state;
                statem[0]=state[0]+1; ttt1=std::sqrt((statem[0])/2.0);
                statem[1]=state[1]+1; ttt2=std::sqrt((statem[1])/2.0);
                E+=-ttt1*ttt2*lambda*Wave.psi(statem).real()/Wave.psi(state).real();
                


                
            }
            else if ((state[0]<0.1)&&(state[1]>=0.1))
            {   statem=state;
                statem[0]=state[0]+1; ttt1=std::sqrt((statem[0])/2.0);
                statem[1]=state[1]+1; ttt2=std::sqrt((statem[1])/2.0);
                
                E+=-ttt1*ttt2*lambda*Wave.psi(statem).real()/Wave.psi(state).real();
                
                statem[1]=state[1]-1; ttt2=std::sqrt((state[1])/2.0);
                
                E+=-ttt1*ttt2*lambda*Wave.psi(statem).real()/Wave.psi(state).real();
                
                
            }
            
            else if ((state[0]>=0.1)&&(state[1]<0.1))
            {   statem=state;
                statem[1]=state[1]+1; ttt2=std::sqrt((statem[1])/2.0);
                statem[0]=state[0]+1; ttt1=std::sqrt((statem[0])/2.0);
                E+=-ttt1*ttt2*lambda*Wave.psi(statem).real()/Wave.psi(state).real();
                
                statem[0]=state[0]-1; ttt1=std::sqrt((state[0])/2.0);
                
                E+=-ttt1*ttt2*lambda*Wave.psi(statem).real()/Wave.psi(state).real();
            }
            else if ((state[0]>=0.1)&&(state[1]>=0.1))
            {   statem=state;
                statem[0]=state[0]+1; ttt1=std::sqrt((statem[0])/2.0);
                statem[1]=state[1]+1; ttt2=std::sqrt((statem[1])/2.0);
                E+=-ttt1*ttt2*lambda*Wave.psi(statem).real()/Wave.psi(state).real();

                
                statem[0]=state[0]-1; ttt1=std::sqrt((state[0])/2.0);
                statem[1]=state[1]+1; ttt2=std::sqrt((statem[1])/2.0);
                E+=-ttt1*ttt2*lambda*Wave.psi(statem).real()/Wave.psi(state).real();

                
                
                statem[0]=state[0]+1; ttt1=std::sqrt((statem[0])/2.0);
                statem[1]=state[1]-1; ttt2=std::sqrt((state[1])/2.0);
                E+=-ttt1*ttt2*lambda*Wave.psi(statem).real()/Wave.psi(state).real();


                statem[0]=state[0]-1; ttt1=std::sqrt((state[0])/2.0);
                statem[1]=state[1]-1; ttt2=std::sqrt((state[1])/2.0);
                E+=-ttt1*ttt2*lambda*Wave.psi(statem).real()/Wave.psi(state).real();
       
                
            }
            */
            
            energy+=E;
            
            Wave.calvp(state,para);
            for(int i=0;i<st;i++)
            {
                o(i,0)+=para[i];
                op(i,0)+=std::conj(para[i]);
                ep(i,0)+=E*std::conj(para[i]);
                for(int j=0;j<st;j++)
                { opo(i,j)+=std::conj(para[i])*para[j];}
            }
        }
        //std::cout<<accept<<std::endl;
        energy/=nsweeps;
        o/=nsweeps;
        op/=nsweeps;
        ep/=nsweeps;
        opo/=nsweeps;
        
        
        std::cout<<energy.real()<<std::endl;
     

        std::complex<double> tempc(double((100*std::pow(0.9,loopopt+1)>0.0001)?(100*std::pow(0.9,loopopt+1)):0.0001),0.0);
        std::complex<double> tempd=0.00;
        for(int i=0;i<st;i++)
        {   F(i,0)=ep(i,0)-energy*op(i,0);
            
            for(int j=0;j<st;j++)
            { S(i,j)=opo(i,j)-op(i,0)*o(j,0);}
            
            tempd=tempc*S(i,i);
            
            S(i,i)+=tempd;
        }


        Matrix<std::complex<double>, Dynamic, 1> dd;
        dd.resize(st,1);

        dd = -0.2*pseudoInverse(S)*F;

        std::vector<std::complex<double> > da_;
        std::vector<std::complex<double> > db_;
        std::vector<std::vector<std::complex<double> >> dc_;

        da_.resize(s);
        db_.resize(s);
        dc_.resize(s,std::vector<std::complex<double> > (sm));
        

        for(int j=0;j<s;j++)
        {
            da_[j]=dd(j,0);
        }
        for(int j=0;j<s;j++)
        {
            db_[j]=dd(j+s,0);
        }
        for(int i=0;i<s;i++)
        {
            for(int j=0;j<sm;j++)
            {
                dc_[i][j]=dd(j+i*sm+s*2,0);
            }

        }

        
        Wave.UpdateParameters(da_,db_,dc_);
        std::flush(std::cout);
        

    }
    

    std::vector<double> sttt(2);
    for(int pp1=0;pp1<nmax;pp1++) {
         for(int pp2=0;pp2<nmax;pp2++) {
        
             sttt[0]=pp1;
             sttt[1]=pp2;
             myfile<<Wave.psi(sttt).real()<<"\t";
             //std::cout<<pp1<<" "<<pp2<<" "<<Wave.psi(sttt).real()<<"\t";
         }
        myfile<<std::endl;
    }
  
  }


