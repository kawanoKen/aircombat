// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
/*
Based on [Ekker 1994] Appendix. B, but modified a bit.
(1) All values should be provided in MKS units. (angles should be in rad.)
(2) Transonic region (M=0.95 to 1.2) is also calculated by same method as either sub/supersonic region to simplify, although it is not so accuarate.
*/
#include "EkkerMDT.h"
#include <string>
#include <map>
#include <boost/math/special_functions/ellint_2.hpp>
#include "MathUtility.h"
#include "Units.h"

namespace py=pybind11;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

using namespace util;

namespace EkkerMDT{
    double areapogv(double d,double ln){
        double R=d/4+ln*ln/d;
        double a=R-d/2;
        return ln*sqrt(R*R-ln*ln)+R*R*asin(ln/R)-2*a*ln;
    }
    double bdycla(double d,double l,double ln,double Mach,double alpha){
        double Sref=M_PI*d*d/4;
        double cdc=sscfdccc(alpha,Mach);
        double Ap=d*(l-ln)+areapogv(d,ln);
        double n;
        if(Mach<=1){
            n=dragfctr(l/d);
        }else{
            n=1.0;
        }
        return k2mk1(l/d)*2+n*Ap*cdc*alpha/Sref;
    }
    double bigkbwa(double M,double A,double r,double s){
        double b=sqrt(abs(M*M-1));
        if(b*A<=0){
            return bigkbwsb(r,s);
        }else{
            if(b*A/4<1){
                return bkbwsba(M,A,r,s);
            }else{
                return bkbwspa(M,A,r,s);
            }
        }
    }
    double bigkbwna(double M,double A,double r,double s){
        double b=sqrt(abs(M*M-1));
        if(b*A<=0){
            return bigkbwsb(r,s);
        }else{
            if(b*A/4<1){
                return bkbwsbna(M,A,r,s);
            }else{
                return bkbwspna(M,A,r,s);
            }
        }
    }
    double bigkbwsb(double r,double s){
        double t=r/s;
        double Kbw;
        if(t>=1){
            Kbw=2.;
        }else{
            double t1=1/t;
            Kbw=(1+pow(t,4))*(0.5*atan(0.5*(t1-t))+M_PI/4)-t*t*((t1-t)+2*atan(t));
            Kbw=(pow(1-t*t,2)-2*Kbw/M_PI)/pow(1-t,2);
        }
        return Kbw;
    }
    double bigkwb(double r,double s){
        double t=r/s;
        double Kbw;
        if(t>1){
            Kbw=2.;
        }else{
            double t1=1/t;
            Kbw=(1+pow(t,4))*(0.5*atan(0.5*(t1-t))+M_PI/4)-t*t*((t1-t)+2*atan(t));
            Kbw=2*Kbw/M_PI/pow(1-t,2);
        }
        return Kbw;
    }
    double bkbwsba(double M,double A,double r,double s){
        double b=sqrt(abs(M*M-1));
        double t=r/s;
        double z=b*A/4;
        double m=sqrt(1-z*z);
        double E=boost::math::ellint_2(m);
        double t1=2*(1+z)*t/(1-t)+1;
        double t2=pow(z/(z+1),2)/2;
        double Kbw=2*pow(z*t/(1-t),2)*atanh(sqrt(1/t1))+pow(b*A/(b*A+4),2);
        Kbw=t2*sqrt(t1)*(t1+1)-Kbw;
        Kbw=8*E*Kbw/pow(z*M_PI,2);
        return Kbw;
    }
    double bkbwsbna(double M,double A,double r,double s){
        double b=sqrt(abs(M*M-1));
        double t=r/s;
        double z=b*A/4;
        double m=sqrt(1-z*z);
        double E=boost::math::ellint_2(m);
        double t2=z+1;
        double F=2*(1-t)/t/A;
        double Kbw;
        if(b<F){
            double t3=(t+1)/2/t;
            double t4=pow((1-t)/t,2);
            double t1=2*(1-t)/b/A/t;
            Kbw=atanh(sqrt(z*(t1-1)/t3))*t2/sqrt(z)+t1*t1*z*sqrt(z);
            Kbw=t4*t2*(atan(sqrt(1/z))-atan(sqrt((t1-1)/t3)))/b/A-Kbw;
            Kbw=Kbw+t3*sqrt((t1-1)*t3);
            Kbw=16*sqrt(z)*E*Kbw/pow(M_PI,2)/t2/t4;
        }else{
            double z1=sqrt(z);
            Kbw=(8*E/pow(M_PI,2)/(1/t-1))*(z1*atan(1/z1)-z/t2);
        }
        return Kbw;
    }
    double bkbwspa(double M,double A,double r,double s){
        double b=sqrt(abs(M*M-1));
        double t=r/s;
        double z=b*A/4;
        double t1=2*(1+z)*t/(1-t)+1;
        double t2=sqrt(z*z-1);
        double t3=1+z;
        double t4=b*A*t3*t/2/(1-t);
        double Kbw=t2*b*A*pow(t/(1-t),2)*acosh(1+2*(1-t)/b/A/t)+t2/t3+z*acos(1/z)/t3;
        Kbw=t2*sqrt(1+b*A*t/(1-t))/t3-Kbw;
        Kbw=(b*A/(b*A+4))*t1*t1*acos((1+t4)/(z+t4))+Kbw;
        Kbw=Kbw/M_PI/t2;
        return Kbw;
    }
    double bkbwspna(double M,double A,double r,double s){
        double b=sqrt(abs(M*M-1));
        double t=r/s;
        double z=b*A/4;
        double t1=(1-t)/t;
        double t2=sqrt(z*z-1);
        double F=2*t1/A;
        double Kbw;
        if(b<F){
            Kbw=t2*acosh(2*t1/b/A);
            Kbw=Kbw+t1*t1*acos(1/z)/4;
            Kbw=t1*t1*t2*asin(b*A/t1/2)/b/A-Kbw;
            Kbw=pow((t+1)/2/t,2)*acos((t*A*b/2+4*(1-t)/A/b)/(t+1))+Kbw;
            Kbw=b*A*Kbw/M_PI/t2/pow(t1,2);
        }else{
            Kbw=(2*z/M_PI/(1/t-1))*(M_PI/2-z*acos(1/z)/t2);
        }
        return Kbw;
    }
    double bsdrgcf(double Mach){
        Eigen::VectorXd p=(Eigen::Matrix<double,9,1>()<<
            -3.330968357195938e4,
            2.103304015137906e5,
            -5.784664805761514e5,
            9.050874924571224e5,
            -8.811731632140930e5,
            5.466276717656434e5,
            -2.109994072780218e5,
            4.633522170518622e4,
            -4.431983251061754e3).finished();
        return polyval(p,Mach);
    }
    double bsdrgsp(double Mach){
        static const Eigen::VectorXd p=(Eigen::Matrix<double,9,1>()<<
            -2.903303569625142e-5,
            8.917615745536690e-4,
            -1.171998069776006e-2,
            8.580630708865064e-2,
            -3.811230276820538e-1,
            1.044387398043697e0,
            -1.697460661376502e0,
            1.405711843047642e0,
            -2.362628974278631e-1).finished();
        return -polyval(p,Mach);
    }
    double cdbbody(double d,double l,double ln,double Mach,double atr){
        double Sref=M_PI*d*d/4;
        double cdc=sscfdccc(atr,Mach);
        double Ap=d*(l-ln)+areapogv(d,ln);
        double n;
        if(Mach<=1){
            n=dragfctr(l/d);
        }else{
            n=1.0;
        }
        return atr*atr*(2*k2mk1(l/d)+n*cdc*Ap*atr/Sref);
    }
    double cdlcomp(double bw,double Sw,double Mach){
        double beta=sqrt(Mach*Mach-1);
        double Aw=bw*bw/Sw;
        static const Eigen::VectorXd p=(Eigen::Matrix<double,8,1>()<<
            -1.131676150597421e0,
            7.455524067331654e0,
            -1.959790137335054e1,
            2.617725155474472e1,
            -1.909733706084593e1,
            8.093378527305321e0,
            -1.217830853307086e0,
            5.365069933169143e-1).finished();
        return polyval(p,beta*Aw/4);
    }
    double cdlwing(double bw,double Sw,double Mach_,double atr){
        double Aw=bw*bw/Sw;
        double Mach=Mach_>0?Mach_:0.0001;
        if(atr==0){
            return 0.0;
        }
        if(Mach<=1){
            double kappaw=0.85;
            double lc2w=atan(2/Aw);
            double CLaW=clawsub(Mach,Aw,lc2w,kappaw);
            return CLaW*atr*atr/1.1;
        }else{
            double betaw=sqrt(Mach*Mach-1);
            if(betaw*Aw/4>=1){
                return 4*atr*atr/betaw;
            }else{
                double corr=cdlcomp(bw,Sw,Mach);
                double CLaW=clawsup(Mach,Aw);
                return 3*pow(CLaW*atr,2)*corr/M_PI/Aw;
            }
        }
    }
    double cdobody(double d,double l,double ln,double Mach,double alt,double nf,bool pwroff){
        double Sref=M_PI*d*d/4.;
        double Swet=surfogv1(d,ln)+(l-ln)*M_PI*d;
        double cdo,CDF;
        if(Mach<=0){
            cdo=0.0;
        }else if(Mach<=1){
            double Mach1=std::min(0.6,Mach);
            double cf=nf*cfturbfp(l,Mach1,alt);
            CDF=cf*Swet/Sref;
            double CDP=cf*(60/pow(l/d,3)+0.0025*l/d)*Swet/Sref;
            cdo=CDF+CDP;
        }else{
            double cf=nf*cfturbfp(l,Mach,alt);
            double CDW=wvdrgogv(d,ln,Mach);
            double CDF=cf*Swet/Sref;
            double CDP;
            if(Mach<1.2){
                double cf=nf*cfturbfp(l,0.6,alt);
                double CDPp6=cf*(60/pow(l/d,3)+0.0025*l/d)*Swet/Sref;
                CDP=CDPp6*(1.2-Mach)/0.2;
            }else{
                CDP=0;
            }
            cdo=CDF+CDW+CDP;
        }
        if(pwroff){
            double cdb;
            if(Mach<=0){
                cdb=0;
            }else if(Mach<=1){
                cdb=0.029/sqrt(CDF);
                if(Mach>0.6){
                    cdb=cdb+bsdrgcf(Mach);
                }
            }else{
                cdb=-bsdrgsp(Mach);
            }
            cdo=cdo+cdb;
        }
        return cdo;
    }
    double cdowbt(double d,double l,double ln,double bw,double Sw,double thickw,double bt,double St,double thickt,double Mach,double alt,double nf,bool pwroff){
        double Sref=M_PI*d*d/4.;
        if(Mach<=0.95 || Mach>=1.1){
            double cdob=cdobody(d,l,ln,Mach,alt,nf,pwroff);
            double cdow=cdowing(bw,Sw,thickw,Mach,alt,nf);
            double cdot=cdowing(bt,St,thickt,Mach,alt,nf);
            return cdob+2*cdow*Sw/Sref+2*cdot*St/Sref;
        }else{
            double cdop95=cdowbt(d,l,ln,bw,Sw,thickw,bt,St,thickt,0.95,alt,nf,pwroff);
            double cdo1p1=cdowbt(d,l,ln,bw,Sw,thickw,bt,St,thickt,1.1,alt,nf,pwroff);
            double dCdM=(cdo1p1-cdop95)/0.15;
            return std::min(cdo1p1,dCdM*(Mach-0.95)+cdop95);
        }
    }
    double cdowing(double bw,double Sw,double thick,double Mach_,double alt,double nf){
        double Mach=Mach_==1.0?0.999999:Mach_;
        double MAC=(2.0/3.0)*2*Sw/bw;
        if(Mach==0){
            return 0.0;
        }else if(Mach<=1){
            double cf=nf*cfturbfp(MAC,Mach,alt);
            return 2*cf*(1+1.2*thick+60*pow(thick,4));
        }else{
            double cf=nf*cfturbfp(MAC,Mach,alt);
            double beta=sqrt(Mach*Mach-1.);
            double B=4.;
            double m=bw*bw/Sw/4;
            double cdw;
            if(m*beta>=1){
                cdw=B*thick*thick/beta;
            }else{
                cdw=B*m*thick*thick;
            }
            return 2*cf+cdw;
        }
    }
    double cdtrim(double d,double l,double ln,double bw,double Sw,double thickw,double bt,double St,double thickt,double Mach,double alt,double atr,double dtr,double nf,bool pwroff){
        double Sref=M_PI*d*d/4.;
        double CDoB=cdobody(d,l,ln,Mach,alt,nf,pwroff);
        double CDoW=cdowing(bw,Sw,thickw,Mach,alt,nf);
        double CDoT=cdowing(bt,St,thickt,Mach,alt,nf);
        double CDoWB=CDoB+2*CDoW*Sw/Sref+CDoT*St/Sref;
        double CDLW=cdlwing(bw,Sw,Mach,atr);
        double CDBa=cdbbody(d,l,ln,Mach,atr);
        double CDiWB=CDLW*Sw/Sref+CDBa;
        double Mt=0.95*Mach;
        double CDLT=cdlwing(bt,St,Mt,dtr);
        double CDT=CDoT+CDLT;
        double CLd=cldwbt(d,bt,St,Mt)*Sref/St;
        double CLT=CLd*dtr;
        double nt=pow(Mt/Mach,2);
        double CDTR=(CDT*cos(dtr)+CLT*sin(dtr))*St*nt/Sref;
        return CDoWB+CDiWB+CDTR;
    }
    double cfturbfp(double chardim,double Mach,double alt){
        std::map<std::string,double> atm=atmosphere(alt);
        double Rn=chardim*Mach*atm["a"]/atm["nu"];
        double cfi=0.455/pow(log10(Rn),2.58);
        double cf;
        if(Mach<=1){
            cf=cfi/(1+0.08*Mach*Mach);
        }else{
            cf=cfi/pow(1+0.144*Mach*Mach,0.65);
        }
        return cf;
    }
    std::pair<double,double> clacma(double d,double l,double ln,double lcg,double bw,double Sw,double lw,double bt,double St,double Mach,double alpha){
        //Combined CLAMSLSB, CLAMSLSP, CLAWBT, CMASUB, CMASUP, and CMAWBT in a single function
        double Sref=M_PI*d*d/4.;
        //Body
        double CLaN=bdycla(d,l,ln,Mach,alpha);
        double cpn,xn;
        if(Mach<1){
            cpn=cpogvsb(ln,d);
        }else{
            cpn=cpogvemp(d,ln,Mach);
        }
        xn=lcg-cpn;
        double CMaNd=xn*CLaN;
        //Wing
        double Aw=bw*bw/Sw;
        double sw=0.5*(d+bw);
        double crw=2*Sw/bw;
        double CLaW,CLaW0,Kbw,Kwb,xbw,xwb;
        if(Mach<1){
            double lc2w=atan(2/Aw);
            double kappaw=0.85;
            CLaW=clawsub(Mach,Aw,lc2w,kappaw);
            CLaW0=clawsub(0,Aw,lc2w,kappaw);
            Kbw=bigkbwsb(d/2,sw);
            Kwb=bigkwb(d/2,sw);
            xbw=lw+xcrbwsub(Mach,Aw,d/2,sw)*crw-lcg;
            xwb=lw+xcrwbsub(Mach,Aw)*crw-lcg;
        }else{
            CLaW=clawsup(Mach,Aw);
            Kbw=bigkbwa(Mach,Aw,d/2,sw);
            Kwb=bigkwb(d/2,sw);
            double betaw=sqrt(Mach*Mach-1);
            double xcrwb=xcrwba(d/2,sw);
            double xcrbw;
            if(betaw*Aw>=0){
                xcrbw=xcrbwabh(Mach,d/2,crw);
            }else{
                xcrbw=xcrbwabl(Mach,Aw,d/2,sw);
            }
            xbw=lw+xcrbw*crw-lcg;
            xwb=lw+xcrwb*crw-lcg;
        }
        //Tail
        double Mt=0.95*Mach;
        double nt=pow(Mt/Mach,2);
        double At=bt*bt/St;
        double st=0.5*(d+bt);
        double crt=2*St/bt;
        double CLaT,Kbt,Ktb,xbt,xtb;
        if(Mt<1){
            double lc2t=atan(2/At);
            double kappat=0.85;
            CLaT=clawsub(Mt,At,lc2t,kappat);
            Kbt=bigkbwsb(d/2,st);
            Ktb=bigkwb(d/2,st);
        }else{
            CLaT=clawsup(Mt,At);
            Kbt=bigkbwna(Mt,At,d/2,st);
            Ktb=bigkwb(d/2,st);
        }
        if(Mt<1){
            double lt=l-crt;
            xbt=lt+xcrbwsub(Mt,At,d/2,st)*crt-lcg;
            xtb=lt+xcrwbsub(Mt,At)*crt-lcg;
        }else{
            double lt=l-crt;
            double betat=sqrt(abs(Mt*Mt-1));
            double xcrtba=xcrwba(d/2,st);
            double xcrbta;
            if(betat*At>=0){
                xcrbta=xcrbwnab(Mt,d/2,crt);
            }else{
                xcrbta=xcrbwabl(Mt,At,d/2,st);
            }
            xbt=lt*xcrbta*crt-lcg;
            xtb=lt*xcrtba*crt-lcg;
        }
        //Downwash
        double deda;
        if(Mach<1){
            double lH=l-St/bt-(lw+Sw/bw);
            double lc4w=atan(3/Aw);
            deda=dedasub(Aw,bw,lc4w,0,lH,0)*CLaW/CLaW0;
        }else{
            double SWPwing=atan(2*crw/bw);
            double lt=l-crt;
            double x0=(lt-(lw+crw))/crw;
            deda=Kwb*dedasup(Mach,SWPwing,x0);
        }
        double cla=CLaN+(Kbw+Kwb)*CLaW*Sw/Sref+(Kbt+Ktb)*CLaT*(1-deda)*nt*St/Sref;
        double cmad=CMaNd-(Kbw*xbw+Kwb*xwb)*CLaW*Sw/Sref-(Kbt*xbt+Ktb*xtb)*CLaT*(1-deda)*nt*St/Sref;
        return std::make_pair(cla,cmad/d);
    }
    double clawsub(double M,double A,double lc2,double kappa){
        double b=sqrt(abs(M*M-1));
        return 2*M_PI*A/(2+sqrt(pow(A*b/kappa,2)*(1+pow(tan(lc2)/b,2))+4));
    }
    double clawsup(double M,double A){
        double b=sqrt(abs(M*M-1));
        if(b*A/4>=1){
            return 4/b;
        }else{
            double E=boost::math::ellint_2(sqrt(1-pow(b*A/4,2)));
            return M_PI*A/2/E;
        }
    }
    std::pair<double,double> cldcmd(double d,double l,double ln,double lcg,double bt,double St,double Mach){
        //Combined CLDWBT and CMDWBT in a single function
        double Mt=0.95*Mach;
        double nt=pow(Mt/Mach,2);
        double Sref=M_PI*d*d/4;
        double At=bt*bt/St;
        double st=0.5*(d+bt);
        double crt=2*St/bt;
        double lt=l-crt;
        double kbt=smlkbw(d/2,st);
        double ktb=smlkwb(d/2,st);
        double CLaT,xcrbt,xcrtb;
        if(Mt<1){
            double lc2t=atan(2/At);
            double kappat=0.85;
            CLaT=clawsub(Mt,At,lc2t,kappat);
            xcrbt=xcrbwsub(Mt,At,d/2,st);
            xcrtb=xcrwbsub(Mt,At);
        }else{
            CLaT=clawsup(Mt,At);
            double betat=sqrt(Mt*Mt-1);
            if(betat*At<0){
                xcrbt=xcrbwabl(Mt,At,d/2,st);
            }else{
                xcrbt=xcrbwnab(Mt,d/2,crt);
            }
            xcrtb=xcrwbd(d/2,st);
        }
        double lbt=lt+xcrbt*crt;
        double ltb=lt+xcrtb*crt;
        double xbt=lbt-lcg;
        double xtb=ltb-lcg;
        double cld=CLaT*nt*(kbt+ktb)*St/Sref;
        double cmd=-CLaT*nt*(kbt*xbt+ktb*xtb)*St/Sref/d;
        return std::make_pair(cld,cmd);
    }
    double cldwbt(double d,double bt,double St,double Mach){
        double Mt=0.95*Mach;
        double nt=pow(Mt/Mach,2);
        double Sref=M_PI*d*d/4;
        double At=bt*bt/St;
        double st=0.5*(d+bt);
        double CLaT;
        if(Mt<1){
            double lc2t=atan(2/At);
            double kappat=0.85;
            CLaT=clawsub(Mt,At,lc2t,kappat);
        }else{
            CLaT=clawsup(Mt,At);
        }
        double kbt=smlkbw(d/2,st);
        double ktb=smlkwb(d/2,st);
        return CLaT*nt*(kbt+ktb)*St/Sref;
    }
    double cpogvemp(double d,double ln,double M){
        double sigma=2*atan(d/2/ln)*180/M_PI;
        double P=(0.083+0.096/pow(M,2))*pow(sigma/10,1.69);
        return 0.5*(50*(M+18)+7*M*M*P*(5*M-18))*ln/(40*(M+18)+7*M*M*P*(4*M-3));
    }
    double cpogvsb(double ln,double d){
        double Vn=vologv1(ln,d);
        return ln-4*Vn/M_PI/pow(d,2);
    }
    double dedasub(double A,double b,double lc4,double lam,double lH,double hH){
        double KA=1./A-1./(1.+pow(A,1.7));
        double Kl=(10.-3.*lam)/7.;
        double KH=(1-abs(hH/b))/pow(2*lH/b,1./3.);
        return 4.44*pow(KA*Kl*KH*sqrt(cos(lc4)),1.19);
    }
    double dedasup(double M,double lam,double x0){
        double psi=M_PI/2-lam;
        double t0=tan(psi)*sqrt(M*M-1);
        double e2,e4,e6,e8;
        if(x0<1.0){
            double e2=0.81;
        }else if(x0<=1.3){
            static const Eigen::VectorXd p21=(Eigen::Matrix<double,4,1>()<<
                -1.666666666666488e0,
                5.999999999999325e0,
                -6.783333333332485e0,
                3.259999999999646e0).finished();
            e2=polyval(p21,x0);
        }else if(x0<2.3){
            static const Eigen::VectorXd p22=(Eigen::Matrix<double,9,1>()<<
                -1.279239755663028e2,
                1.843272381677282e3,
                -1.155286798315866e4,
                4.113252460613160e4,
                -9.098009805564612e4,
                1.280057606805492e5,
                -1.118657697776747e5,
                5.551390292092566e4,
                -1.197584408390502e4).finished();
            e2=polyval(p22,x0);
        }else{
            e2=0.97;
        }
        if(x0<1.0){
            e4=0.65;
        }else if(x0<=1.6){
            static const Eigen::VectorXd p41=(Eigen::Matrix<double,5,1>()<<
                -4.166666666665245e0,
                2.083333333332625e1,
                -3.845833333332031e1,
                3.134166666665624e1,
                -8.899999999996956e0).finished();
            e4=polyval(p41,x0);
        }else if(x0<=3.0){
            static const Eigen::VectorXd p42=(Eigen::Matrix<double,9,1>()<<
                4.475474425067136e0,
                -8.198016093532951e1,
                6.529374943388477e2,
                -2.952920552556513e3,
                8.292996506025491e3,
                -1.480813974021849e4,
                1.641631929076350e4,
                -1.032958858524409e4,
                2.825103492654550e3).finished();
            e4=polyval(p42,x0);
        }else{
            e4=0.87;
        }
        if(x0<1.0){
            e6=0.54;
        }else if(x0<=1.4){
            static const Eigen::VectorXd p61=(Eigen::Matrix<double,4,1>()<<
                1.600881353985564e-14,
                -6.387361790702744e-14,
                1.000000000000832e-1,
                4.399999999999644e-1).finished();
            e6=polyval(p61,x0);
        }else if(x0<=1.8){
            static const Eigen::VectorXd p62=(Eigen::Matrix<double,5,1>()<<
                -3.333333333309070e1,
                2.116666666651215e2,
                -5.026666666629868e2,
                5.294833333294496e2,
                -2.082299999984673e2).finished();
            e6=polyval(p62,x0);
        }else if(x0<3.6){
            static const Eigen::VectorXd p63=(Eigen::Matrix<double,7,1>()<<
                6.356470889426016e-3,
                -5.240723295023863e-2,
                -2.580914111953288e-2,
                1.556718856871693e0,
                -6.218769329470563e0,
                1.005756492417748e1,
                -5.269710040962651e0).finished();
            e6=polyval(p63,x0);
        }else{
            e6=0.79;
        }
        if(x0<1.0){
            e8=0.44;
        }else if(x0<=1.4){
            static const Eigen::VectorXd p81=(Eigen::Matrix<double,5,1>()<<
                -1.666666666665405e1,
                7.999999999993737e1,
                -1.433333333332168e2,
                1.136499999999038e2,
                -3.320999999997028e1).finished();
            e8=polyval(p81,x0);
        }else if(x0<=2.2){
            static const Eigen::VectorXd p82=(Eigen::Matrix<double,9,1>()<<
                -8.680554787360018e2,
                1.252976079851500e4,
                -7.886457638672026e4,
                2.827083084908162e5,
                -6.312798925797358e5,
                8.991348588842052e5,
                -7.977027436345048e5,
                4.030393822169418e5,
                -8.878791230326184e4).finished();
            e8=polyval(p82,x0);
        }else if(x0<3.4){
            static const Eigen::VectorXd p83=(Eigen::Matrix<double,9,1>()<<
                3.866380123250028e0,
                -9.169392757855618e1,
                9.469642944017384e2,
                -5.562064279886374e3,
                2.032063141263390e4,
                -4.728372988378842e4,
                6.842861819191089e4,
                -5.630916695234881e4,
                2.017213892626900e4).finished();
            e8=polyval(p83,x0);
        }else{
            e8=0.7;
        }
        double dedaw;
        if(t0<=0.2){
            dedaw=(e2-e4)*(0.2-t0)/0.2+e2;
        }else if(t0<=0.4){
            dedaw=(e2-e4)*(0.4-t0)/0.2+e4;
        }else if(t0<=0.6){
            dedaw=(e4-e6)*(0.6-t0)/0.2+e6;
        }else if(t0<=0.8){
            dedaw=(e6-e8)*(0.8-t0)/0.2+e8;
        }else{//if(t0>0.8){
            dedaw=e8-(e6-e8)*(t0-0.8)/0.2;
        }
        return std::max(0.0,dedaw);
    }
    double dragfctr(double fineness){
        static const Eigen::VectorXd p=(Eigen::Matrix<double,8,1>()<<
            -3.610649936485454e-12,
            8.309751611041349e-10,
            -7.347789500313429e-8,
            3.037472628776377e-6,
            -5.190776944854130e-5,
            -1.773312264112830e-4,
            2.135202437974324e-2,
            5.187105185342232e-1).finished();
        return polyval(p,fineness);
    }
    double k2mk1(double fineness){
        static const Eigen::VectorXd pa=(Eigen::Matrix<double,8,1>()<<
            -6.3424e-7,
            3.1805e-5,
            -6.0351e-4,
            5.0080e-3,
            -9.9444e-3,
            -1.1439e-1,
            7.7981e-1,
            -6.6077e-1).finished();
        static const Eigen::VectorXd pb=(Eigen::Matrix<double,2,1>()<<1.6667e-3,9.4667e-1).finished();
        if(fineness<0){
            fineness=0;
        }
        double amf;
        if(fineness<14){
            amf=polyval(pa,fineness);
        }else{
            amf=polyval(pb,fineness);
        }
        return std::max(0.0,amf);
    }
    double smlkbw(double r,double s){
        return bigkwb(r,s)-smlkwb(r,s);
    }
    double smlkwb(double r,double s){
        double t=s/r;
        double kwb;
        if(t<=1){
            kwb=1;
        }else{
            kwb=pow(M_PI*(t+1)/t,2)/4+M_PI*pow((t*t+1)/(t*(t-1)),2)*asin((t*t-1)/(t*t+1));
            kwb=kwb-2*M_PI*(t+1)/t/(t-1)+pow((t*t+1)/(t*(t-1)),2)*pow(asin((t*t-1)/(t*t+1)),2);
            kwb=kwb-4*(t+1)*asin((t*t-1)/(t*t+1))/t/(t-1)+8*log((t*t+1)/2/t)/pow(t-1,2);
            kwb=kwb/pow(M_PI,2);
        }
        return kwb;
    }
    double sscfdccc(double alpha,double M){
        static const Eigen::VectorXd pa=(Eigen::Matrix<double,5,1>()<<-2.5513,4.9961,-1.5000,0.1394,1.1967).finished();
        static const Eigen::VectorXd pb=(Eigen::Matrix<double,6,1>()<<-2.1943e2,9.0057e2,-1.4619e3,1.1695e3,-4.5889e2,7.1919e1).finished();
        static const Eigen::VectorXd pc=(Eigen::Matrix<double,5,1>()<<154.7931,-553.2488,737.6712,-434.1301,96.7047).finished();
        static const Eigen::VectorXd pd=(Eigen::Matrix<double,6,1>()<<-3.4916,7.9077,-6.1199,2.3609,-7.8519e-2,1.2400).finished();
        double k=sin(alpha)*M;
        if(k>1){
            k=1/k;
            if(k>0.8){
                return polyval(pc,k);
            }else{
                return polyval(pd,k);
            }
        }else{
            if(k>0.6){
                return polyval(pb,k);
            }else{
                return polyval(pa,k);
            }
        }
    }
    double surfogv1(double diameter,double length){
        double ldr=length/diameter;
        double ldr2=ldr*ldr;
        double surf1=asin(ldr/(ldr2+0.25));
        return M_PI*pow(diameter,2)*(pow(ldr2+0.25,2)*surf1-ldr*(ldr2-0.25));
    }
    double vologv1(double logv,double dbase){
        double l=logv;
        double d=dbase;
        double R=d/4+l*l/d;
        double vol=(R-d/2)*(l*sqrt(R*R-l*l)+R*R*asin(l/R));
        vol=M_PI*(l*(2*R*R-R*d+d*d/4)-pow(l,3)/3-vol);
        return vol;
    }
    double wvdrgogv(double diameter,double length,double Mach){
        double sigma=2*(180/M_PI)*atan(diameter/2/length);
        double P=(0.083+0.096/(Mach*Mach))*pow(sigma/10,1.69);
        double lod2=pow(length/diameter,2);
        return P*(1-(392*lod2-32)/28/lod2/(Mach+18));
    }
    double xcrbwabh(double M,double r,double cr){
        //supersonic, with afterbody and low aspect ratio
        double b=sqrt(abs(M*M-1));
        double t=2*b*r/cr;
        static const Eigen::VectorXd p=(Eigen::Matrix<double,4,1>()<<1.7611e-2,-1.0474e-1,5.9054e-1,5.0298e-1).finished();
        return polyval(p,t);
    }
    double xcrbwabl(double M,double A,double r,double s){
        //supersonic, with afterbody and high aspect ratio
        double b=sqrt(abs(M*M-1));
        static const Eigen::VectorXd p=(Eigen::Matrix<double,4,1>()<<1.7976e0,-8.2857e-1,4.3667e-1,-5.7381e-17).finished();
        double t1=b*A;
        double t2=r/s;
        double slope=polyval(p,t2);
        return slope*t1+0.5;
    }
    double xcrbwnab(double M,double r,double cr){
        //supersonic and without afterbody
        double b=sqrt(abs(M*M-1));
        double t=2*b*r/cr;
        static const Eigen::VectorXd p=(Eigen::Matrix<double,8,1>()<<8.3600e0,-3.0112e1,4.3283e1,-3.1734e1,1.2746e1,-3.0908e0,7.1424e-1,5.0019e-1).finished();
        if(t>0.9){
            return 2./3.;
        }else{
            return polyval(p,t);
        }
    }
    double xcrbwsub(double M,double A,double r,double s){
        //subsonic
        double b=sqrt(abs(1-M*M));
        double t=r/s;
        double rs6,rs4,rs2,rs0;
        if(b*A<4){
            static const Eigen::VectorXd p6=(Eigen::Matrix<double,5,1>()<<
                -1.503003360588808e-4,
                1.500533636756856e-3,
                -5.843498893035511e-3,
                1.266301279846288e-2,
                4.997093223254211e-1).finished();
            rs6=polyval(p6,b*A);
            static const Eigen::VectorXd p4=(Eigen::Matrix<double,10,1>()<<
                1.439562405737517e-4,
                -2.577574370693720e-3,
                1.915260292425828e-2,
                -7.636557567970656e-2,
                1.764081283046741e-1,
                -2.379409374985050e-1,
                1.805435508411277e-1,
                -7.176958947911805e-2,
                5.130816740444732e-3,
                4.999682326019724e-1).finished();
            rs4=polyval(p4,b*A);
            static const Eigen::VectorXd p2=(Eigen::Matrix<double,9,1>()<<
                1.870466827291419e-4,
                -3.055345401484121e-3,
                2.012956338935916e-2,
                -6.808482446989989e-2,
                1.241295486321807e-1,
                -1.160715264335058e-1,
                4.940441636896099e-2,
                -4.087742141651518e-2,
                5.000384633792437e-1).finished();
            rs2=polyval(p2,b*A);
        }else{
            rs6=0.5+2.6*0.05/9;
            rs4=0.45+4*0.05/9;
            rs2=0.4+1.5*0.05/9;
        }
        if(b*A>=7){
            rs0=0.25;
        }else{
            static const Eigen::VectorXd p0=(Eigen::Matrix<double,7,1>()<<
                1.368278964039268e-5,
                -2.870181449710582e-4,
                2.101837326902790e-3,
                -6.106506978595964e-3,
                1.093840736149155e-2,
                -7.485137721564938e-2,
                5.003830336176011e-1).finished();
            rs0=polyval(p0,b*A);
        }
        if(t>0.6){
            return rs6+(rs6-rs4)*(t-0.6)/0.2;
        }else if(t>0.4){
            return rs4+(rs6-rs4)*(t-0.4)/0.2;
        }else if(t>0.2){
            return rs2+(rs4-rs2)*(t-0.2)/0.2;
        }else{
            return rs0+(rs2-rs0)*t/0.2;
        }
    }
    double xcrw(double M,double A){
        double b=sqrt(abs(M*M-1));
        double mb=b*A;
        return 2./3.;
    }
    double xcrwba(double r,double s){
        double t=r/s;
        static const Eigen::VectorXd p=(Eigen::Matrix<double,4,1>()<<-0.0822,0.2009,-0.1190,0.6667).finished();
        return polyval(p,t);
    }
    double xcrwbd(double r,double s){
        double t=r/s;
        static const Eigen::VectorXd p=(Eigen::Matrix<double,8,1>()<<0.8716,-3.0894,4.3106,-3.1344,1.3829,-0.3885,0.0473,0.6669).finished();
        return polyval(p,t);
    }
    double xcrwbsub(double M,double A){
        double b=sqrt(abs(1-M*M));
        if(b*A<2){
            static const Eigen::VectorXd p=(Eigen::Matrix<double,6,1>()<<
                -1.130484330484368e-2,
                5.420357420357615e-2,
                -1.073193473193506e-1,
                1.446257446257462e-1,
                -1.627277907277905e-1,
                6.555089355089354e-1).finished();
            return polyval(p,b*A);
        }else if(b*A>=5){
            return 0.5+3.5/90;
        }else{
            static const Eigen::VectorXd p=(Eigen::Matrix<double,3,1>()<<
                -1.785830762908004e-17,-5.555555555555476e-3,5.666666666666665e-1).finished();
            return polyval(p,b*A);
        }
    }
}

void exportEkkerMDT(py::module &m_){
    using namespace EkkerMDT;

	auto m=m_.def_submodule("EkkerMDT");
    m.def("areapogv",&areapogv);
    m.def("bdycla",&bdycla);
    m.def("bigkbwa",&bigkbwa);
    m.def("bigkbwna",&bigkbwna);
    m.def("bigkbwsb",&bigkbwsb);
    m.def("bigkwb",&bigkwb);
    m.def("bkbwsba",&bkbwsba);
    m.def("bkbwsbna",&bkbwsbna);
    m.def("bkbwspa",&bkbwspa);
    m.def("bkbwspna",&bkbwspna);
    m.def("bsdrgcf",&bsdrgcf);
    m.def("bsdrgsp",&bsdrgsp);
    m.def("cdbbody",&cdbbody);
    m.def("cdlcomp",&cdlcomp);
    m.def("cdlwing",&cdlwing);
    m.def("cdobody",&cdobody);
    m.def("cdowbt",&cdowbt);
    m.def("cdowing",&cdowing);
    m.def("cdtrim",&cdtrim);
    m.def("cfturbfp",&cfturbfp);
    m.def("clacma",&clacma);
    m.def("clawsub",&clawsub);
    m.def("clawsup",&clawsup);
    m.def("cldcmd",&cldcmd);
    m.def("cldwbt",&cldwbt);
    m.def("cpogvemp",&cpogvemp);
    m.def("cpogvsb",&cpogvsb);
    m.def("dedasub",&dedasub);
    m.def("dedasup",&dedasup);
    m.def("dragfctr",&dragfctr);
    m.def("k2mk1",&k2mk1);
    m.def("smlkbw",&smlkbw);
    m.def("smlkwb",&smlkwb);
    m.def("sscfdccc",&sscfdccc);
    m.def("surfogv1",&surfogv1);
    m.def("vologv1",&vologv1);
    m.def("wvdrgogv",&wvdrgogv);
    m.def("xcrbwabh",&xcrbwabh);
    m.def("xcrbwabl",&xcrbwabl);
    m.def("xcrbwnab",&xcrbwnab);
    m.def("xcrbwsub",&xcrbwsub);
    m.def("xcrw",&xcrw);
    m.def("xcrwba",&xcrwba);
    m.def("xcrwbd",&xcrwbd);
    m.def("xcrwbsub",&xcrwbsub);
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
