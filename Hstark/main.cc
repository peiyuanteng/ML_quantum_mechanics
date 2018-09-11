
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


class Wavefunction1d{
    
private:
    //Neural-network
    std::vector<std::complex<double> > a;
    std::vector<std::complex<double> > b;
    std::vector<std::vector<std::complex<double> >> c;

    
    //Number of n and wavefunction
    int n;
    int m;
    

    
public:
    
    Wavefunction1d(int n_,int m_):n(n_),m(m_){
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



class HamiltonianStark
{
private:
    double omega;
    double mass;
    double electric;
public:
    HamiltonianStark(double o, double m, double e):omega(o),mass(m),electric(e){}

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



std::vector<std::complex<double> > run(){

    Wavefunction1d Wave(10,1);
    //std::cout<<Wave.psi(1)<<std::endl;
    
    
    HamiltonianStark H(1,1,0);
    
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

    int nmax=20;

    
    
    int step=100;
    
    std::vector<std::complex<double> > resultenergy(100);
    
    for(int loopopt=0;loopopt<step;loopopt++)
    {
        /*
        for(int i=0;i<s;i++)
        {
            std::cout<<Wave.geta()[i]<<Wave.getb()[i]<<std::endl;
            for(int j=0;j<sm;j++)
            {
                std::cout<<Wave.getc()[i][j]<<std::endl;
            }
        }*/
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
        int flip=0;
        int accept=0;
        std::vector<double> state(1);
        std::vector<double> staten(1);
        std::vector<double> statem(1);
        state[0]=rand()%nmax;
        state=staten;

        for(int n=0;n<nsweeps*0.05;n+=1)
        {
            flip=(rand()%2-0.5)*2;
            staten[0]=state[0]+flip;
            
            if (staten[0]<0) staten[0]=1;
            if (staten[0]>=nmax) staten[0]=nmax-2;
            
            
            std::complex<double> w = Wave.psi(staten) / Wave.psi(state);
            
            if (sample(gen)< std::norm(w)) {
                
                state[0]=staten[0]; accept++;
            }
            
        }

        
  
        accept=0;
        
        
        
        std::complex<double> energy(0,0);

        
        for(int n=0;n<nsweeps;n+=1)
        {
            flip=(disi(gen)-0.5)*2;
            staten[0]=state[0]+flip;

            if (staten[0]<0) staten[0]=1;
            if (staten[0]>=nmax) staten[0]=nmax-2;

            
            std::complex<double> w = Wave.psi(staten) / Wave.psi(state);
            
            if (sample(gen)< std::norm(w)) {
                
                state[0]=staten[0]; accept++;
            }
            
            
            std::complex<double> E(0,0);
            
            E=H.matrixelement(state);
            
            double ttt=0.0;
            if (staten[0]<0.1)
            {statem[0]=state[0]+1; ttt-=std::sqrt((statem[0])/2)*Wave.psi(statem).real()/ Wave.psi(state).real();}
            else
            {statem[0]=state[0]+1; ttt-=std::sqrt((statem[0])/2)*Wave.psi(statem).real()/ Wave.psi(state).real();
            statem[0]=state[0]-1; ttt-=std::sqrt((state[0])/2)*Wave.psi(statem).real()/ Wave.psi(state).real();}
            
            //qE factor
            E+=ttt*1.0;
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
        
        energy/=nsweeps;
        o/=nsweeps;
        op/=nsweeps;
        ep/=nsweeps;
        opo/=nsweeps;
        
        
        std::cout<<energy.real()<<std::endl;
        resultenergy[loopopt]=energy;

        std::complex<double> tempc(double((100*std::pow(0.9,loopopt+1)>0.0001)?(100*std::pow(0.9,loopopt+1)):0.0001),0.0);
       // std::complex<double> tempc=0.01;
        std::complex<double> tempd=0.00;
        for(int i=0;i<st;i++)
        {   F(i,0)=ep(i,0)-energy*op(i,0);
            
            for(int j=0;j<st;j++)
            { S(i,j)=opo(i,j)-op(i,0)*o(j,0);}
            
            tempd=tempc*S(i,i);
            //std::cout<<S(i,i)<<tempc<<std::endl;
            S(i,i)+=tempd;
            
        }
//std::cout<<energy<<std::endl;

        Matrix<std::complex<double>, Dynamic, 1> dd;
        dd.resize(st,1);
        
        dd = -0.1*pseudoInverse(S)*F;

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
    std::vector<double> sttt(1);
    for(int pp=0;pp<nmax;pp++) {sttt[0]=pp;std::cout<<Wave.psi(sttt).real()<<" ";}
    return resultenergy;
}




int main(int argc, char *argv[]){
    
    
    std::ofstream myfile;
    myfile.open ("data.txt");
    std::vector<std::complex<double> > energy(100);
    std::vector<std::complex<double> > temp(100);
    for(int j=0;j<100;j++)
    {
        energy[j]=0;
    }
    for(int i=0;i<1;i++)
    {
    temp=run();
    for(int j=0;j<100;j++)
    {
        energy[j]+=temp[j];
    }
    }


    
    for(int j=0;j<100;j++)
    {
        myfile<<energy[j].real()<<"\t";
    }
    
    myfile.close();
}
