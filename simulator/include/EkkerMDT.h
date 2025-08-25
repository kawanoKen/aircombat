// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "Common.h"
#include <utility>
#include <pybind11/pybind11.h>
/*
Based on [Ekker 1994] Appendix. B, but modified a bit.
(1) All values should be provided in MKS units. (angles should be in rad.)
(2) Transonic region (M=0.95 to 1.2) is also calculated by same method as either sub/supersonic region to simplify, although it is not so accuarate.
*/

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

namespace EkkerMDT{
    double PYBIND11_EXPORT areapogv(double d,double ln);
    double PYBIND11_EXPORT bdycla(double d,double l,double ln,double Mach,double alpha=0.0);
    double PYBIND11_EXPORT bigkbwa(double M,double A,double r,double s);
    double PYBIND11_EXPORT bigkbwna(double M,double A,double r,double s);
    double PYBIND11_EXPORT bigkbwsb(double r,double s);
    double PYBIND11_EXPORT bigkwb(double r,double s);
    double PYBIND11_EXPORT bkbwsba(double M,double A,double r,double s);
    double PYBIND11_EXPORT bkbwsbna(double M,double A,double r,double s);
    double PYBIND11_EXPORT bkbwspa(double M,double A,double r,double s);
    double PYBIND11_EXPORT bkbwspna(double M,double A,double r,double s);
    double PYBIND11_EXPORT bsdrgcf(double Mach);
    double PYBIND11_EXPORT bsdrgsp(double Mach);
    double PYBIND11_EXPORT cdbbody(double d,double l,double ln,double Mach,double atr);
    double PYBIND11_EXPORT cdlcomp(double bw,double Sw,double Mach);
    double PYBIND11_EXPORT cdlwing(double bw,double Sw,double Mach_,double atr);
    double PYBIND11_EXPORT cdobody(double d,double l,double ln,double Mach,double alt,double nf=1.1,bool pwroff=false);
    double PYBIND11_EXPORT cdowbt(double d,double l,double ln,double bw,double Sw,double thickw,double bt,double St,double thickt,double Mach,double alt,double nf=1.1,bool pwroff=false);
    double PYBIND11_EXPORT cdowing(double bw,double Sw,double thick,double Mach_,double alt,double nf=1.1);
    double PYBIND11_EXPORT cdtrim(double d,double l,double ln,double bw,double Sw,double thickw,double bt,double St,double thickt,double Mach,double alt,double atr,double dtr,double nf=1.1,bool pwroff=false);
    double PYBIND11_EXPORT cfturbfp(double chardim,double Mach,double alt);
    std::pair<double,double> PYBIND11_EXPORT clacma(double d,double l,double ln,double lcg,double bw,double Sw,double lw,double bt,double St,double Mach,double alpha=0.0);
    double PYBIND11_EXPORT clawsub(double M,double A,double lc2,double kappa);
    double PYBIND11_EXPORT clawsup(double M,double A);
    std::pair<double,double> PYBIND11_EXPORT cldcmd(double d,double l,double ln,double lcg,double bt,double St,double Mach);
    double PYBIND11_EXPORT cldwbt(double d,double bt,double St,double Mach);
    double PYBIND11_EXPORT cpogvemp(double d,double ln,double M);
    double PYBIND11_EXPORT cpogvsb(double ln,double d);
    double PYBIND11_EXPORT dedasub(double A,double b,double lc4,double lam,double lH,double hH);
    double PYBIND11_EXPORT dedasup(double M,double lam,double x0);
    double PYBIND11_EXPORT dragfctr(double fineness);
    double PYBIND11_EXPORT k2mk1(double fineness);
    double PYBIND11_EXPORT smlkbw(double r,double s);
    double PYBIND11_EXPORT smlkwb(double r,double s);
    double PYBIND11_EXPORT sscfdccc(double alpha,double M);
    double PYBIND11_EXPORT surfogv1(double diameter,double length);
    double PYBIND11_EXPORT vologv1(double logv,double dbase);
    double PYBIND11_EXPORT wvdrgogv(double diameter,double length,double Mach);
    double PYBIND11_EXPORT xcrbwabh(double M,double r,double cr);
    double PYBIND11_EXPORT xcrbwabl(double M,double A,double r,double s);
    double PYBIND11_EXPORT xcrbwnab(double M,double r,double cr);
    double PYBIND11_EXPORT xcrbwsub(double M,double A,double r,double s);
    double PYBIND11_EXPORT xcrw(double M,double A);
    double PYBIND11_EXPORT xcrwba(double r,double s);
    double PYBIND11_EXPORT xcrwbd(double r,double s);
    double PYBIND11_EXPORT xcrwbsub(double M,double A);
}

void exportEkkerMDT(pybind11::module &m);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
