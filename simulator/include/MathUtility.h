/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief 数学的な操作に関するユーティリティ関数を集めたファイル
 */
#pragma once
#include "Common.h"
#include <cmath>
#include <random>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/CXX11/Tensor>
#include "traits/eigen.h"

namespace py=pybind11;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class Time;

ASRC_INLINE_NAMESPACE_BEGIN(util)

double PYBIND11_EXPORT sinc(const double &x,const double &eps=1e-3);
double PYBIND11_EXPORT sincMinusOne(const double &x,const double &eps=1e-3);
double PYBIND11_EXPORT sincMinusOne_x2(const double &x,const double &eps=1e-3);
double PYBIND11_EXPORT oneMinusCos(const double &x,const double &eps=1e-3);
double PYBIND11_EXPORT oneMinusCos_x2(const double &x,const double &eps=1e-3);
double PYBIND11_EXPORT atanx_x(const double &x,const double &eps=1e-3);
double PYBIND11_EXPORT acosOne_OnePlusx2_x(const double &x,const double &eps=1e-3);

/**
 * @brief vに直交するベクトルを一つ生成して返す。
 */
Eigen::Vector3d PYBIND11_EXPORT getOrthogonalVector(const Eigen::Vector3d &v);

/**
 * @brief 任意の3次元ベクトルxに対してMx=v×xとなる歪対称行列Mを返す。
 */
template<typename T>
Eigen::Matrix<T,3,3> skewMatrix(const Eigen::Matrix<T,3,1>& v){
    return Eigen::Matrix<T,3,3>({
        {0,-v(2),v(1)},
        {v(2),0,-v(0)},
        {-v(1),v(0),0}});
}

/**
 * @brief pを係数とするxの多項式の値を返す。
 * 
 * @param [in] p 係数のリスト(先頭が最高次係数、末尾が定数項)
 * @param [in] x 入力変数の値
 */
template<typename T, int _Rows>
T polyval(const Eigen::Matrix<T, _Rows, 1>& p,const T &x){
    Eigen::Index n=p.size();
    T xx=1.0;
    T ret=0;
    for(int i=0;i<n;++i){
        ret+=xx*p(n-1-i);
        xx*=x;
    }
    return ret;
}
/**
 * @brief pを係数とするxの多項式の導関数値を返す。
 * 
 * @param [in] p 係数のリスト(先頭が最高次係数、末尾が定数項)
 * @param [in] x 入力変数の値
 */
template<typename T, int _Rows>
Eigen::Matrix<T,_Rows,1> polyderiv(const Eigen::Matrix<T,_Rows,1>& p,const T &x){
    Eigen::Index n=p.size()-1;
    Eigen::Matrix<T,_Rows,1> ret;
    for(int i=0;i<n;++i){
        ret(i)=p(i)*(n-1-i);
    }
    return ret;
}

/**
 * @brief Tensorの次元数を返す。
 */
template<traits::tensor TensorType>
requires (
    traits::has_dimensions<TensorType>
    || traits::has_sizes<TensorType>
)
auto getTensorDimensions(const TensorType& t){
    if constexpr (traits::has_dimensions<TensorType>){
        return t.dimensions();
    }else{
        return t.sizes();
    }
}

/**
 * @brief Tensorの要素数を返す。
 */
template<traits::tensor TensorType>
requires (
    traits::has_dimensions<TensorType>
    || traits::has_sizes<TensorType>
)
auto getTensorSize(const TensorType& t){
    if constexpr (traits::has_size<TensorType>){
        return t.size();
    }else{
        auto dimensions=getTensorDimensions(t);
        typename Eigen::internal::traits<TensorType>::Index ret=1;
        for(auto d : dimensions){
            ret*=d;
        }
        return ret;
    }
}

/**
 * @brief 1次元テーブルデータを内挿する際に参照すべきインデックスと重み付け係数を返す。
 * 
 * @param [in] x     テーブルのデータ点のリスト(ソート済)
 * @param [in] p     内挿したい点の位置
 * @param [in] clamp trueにすると、p が x の範囲外にある場合、xの端点にclampされる。falseの場合、線形外挿となる。
 * 
 * @returns \f$ p = x(i) + a (x(i+1)-x(i)) \f$ を満たす\f$(i,a)\f$
 */
template<typename VectorType>
requires (
    ( traits::matrix<VectorType> )
    || ( traits::tensor<VectorType>)
)
std::pair<Eigen::Index,typename Eigen::internal::traits<VectorType>::Scalar> _search1d(const VectorType &x,const typename Eigen::internal::traits<VectorType>::Scalar &p, bool clamp=false){
    Eigen::Index lb=0;
    Eigen::Index ub;
    if constexpr (traits::matrix<VectorType>){
        assert(x.rows()==1 || x.cols()==1);
        ub=x.size()-2;
    }else if constexpr (traits::tensor<VectorType>){
        if constexpr (Eigen::internal::traits<VectorType>::NumDimensions>1){
            auto dims=getTensorDimensions(x);
            bool found_more_than_one=false;
            for(auto d: dims){
                if(d>1){
                    if(found_more_than_one){
                        throw std::runtime_error("asrc::core::_search1d failed. x is not 1-dim tensor.");
                    }else{
                        found_more_than_one=true;
                    }
                }
            }
            return _search1d(x.reshape(Eigen::DSizes<Eigen::Index,1>{getTensorSize(x)}),p);
        }
        ub=getTensorSize(x)-2;
    }else{
        ub=x.size()-2;
    }
    if(ub<0){
        // x has only one element
        return {lb,x(lb)};
    }
    if(p<x(lb)){
        if(clamp){
            return {lb,0};
        }else{
            return {lb,(p-x(lb))/(x(lb+1)-x(lb))};
        }
    }
    if(x(ub)<=p){
        if(clamp){
            return {ub,1};
        }else{
            return {ub,(p-x(ub))/(x(ub+1)-x(ub))};
        }
    }
    Eigen::Index i;
    while(ub-lb>1){
        i=(lb+ub)/2;
        if(x(i)<=p){
            lb=i;
        }else{// if(p<x(i)){
            ub=i;
        }
    }
    return {lb,(p-x(lb))/(x(lb+1)-x(lb))};
}

/**
 * @brief N次元テーブルデータをM個の点において線形内挿する。
 * 
 * @param [in] point テーブルの各次元のデータ点のリスト(ソート済)
 * @param [in] table テーブルの値リスト(N次元のテンソル)
 * @param [in] x     内挿したい点の位置(M×N行列)
 * @param [in] clamp trueにすると、x が point の範囲外にある場合、 point の端点にclampされる。falseの場合、線形外挿となる。
 * 
 * @returns 内挿された値を表すM次元ベクトル
 */
template<traits::tensor TensorType, traits::matrix MatrixType, typename Scalar=typename Eigen::internal::traits<TensorType>::Scalar>
requires (
    std::same_as<Scalar,typename Eigen::internal::traits<MatrixType>::Scalar>
)
Eigen::VectorX<Scalar> interpn(const std::vector<Eigen::VectorX<Scalar>>& point,const TensorType &table,const MatrixType &x, bool clamp=false){
    using TensorTraits=Eigen::internal::traits<TensorType>;
    using MatrixTraits=Eigen::internal::traits<MatrixType>;
    constexpr int nDim=TensorTraits::NumDimensions;
    Eigen::Index nx=x.rows();
    Eigen::Index nt=getTensorDimensions(table)[nDim-1];
    Eigen::Index m=getTensorSize(table);
    Eigen::Tensor<Scalar,3,TensorTraits::Layout,typename TensorTraits::Index> store;
    for(int d=nDim-1;d>=0;--d){
        m/=nt;
        Eigen::Tensor<Scalar,2> tmp2(nx,m);
        if(d==nDim-1){
            Eigen::Tensor<Scalar,2,TensorTraits::Layout,typename TensorTraits::Index> first=table.reshape(Eigen::array<Eigen::Index,2>{{m,nt}});
            for(Eigen::Index i=0;i<nx;++i){
                std::pair<Eigen::Index,Scalar> idx=_search1d(point[d],x(i,d),clamp);
                for(Eigen::Index j=0;j<m;++j){
                    tmp2(i,j)=idx.second*(first(j,idx.first+1)-first(j,idx.first))+first(j,idx.first);
                }
            }
        }else{
            for(Eigen::Index i=0;i<nx;++i){
                std::pair<Eigen::Index,Scalar> idx=_search1d(point[d],x(i,d),clamp);
                for(Eigen::Index j=0;j<m;++j){
                    tmp2(i,j)=idx.second*(store(i,j,idx.first+1)-store(i,j,idx.first))+store(i,j,idx.first);
                }
            }
        }
        nt=(d>=1)?getTensorDimensions(table)[d-1]:1;
        store=tmp2.reshape(Eigen::array<Eigen::Index,3>{{nx,m/nt,nt}});
    }
    typedef Eigen::Map<Eigen::VectorX<Scalar>> maptype;
    return maptype(store.data(),nx);
}

/**
 * @brief N次元テーブルデータをM個の点において線形内挿した際の偏導関数値を返す。
 * 
 * @param [in] point テーブルの各次元のデータ点のリスト(ソート済)
 * @param [in] table テーブルの値リスト(N次元のテンソル)
 * @param [in] x     内挿したい点の位置(M×N行列)
 * 
 * @returns 各点の偏導関数値を表すN×M行列(i行目がi次元目の変数による偏導関数値)
 */
template<traits::tensor TensorType, traits::matrix MatrixType, typename Scalar=typename Eigen::internal::traits<TensorType>::Scalar>
requires (
    std::same_as<Scalar,typename Eigen::internal::traits<MatrixType>::Scalar>
)
Eigen::Matrix<Scalar,-1,-1,Eigen::internal::traits<TensorType>::Layout> interpgradn(const std::vector<Eigen::VectorX<Scalar>>& point,const TensorType &table,const MatrixType &x){
    using TensorTraits=Eigen::internal::traits<TensorType>;
    using MatrixTraits=Eigen::internal::traits<MatrixType>;
    constexpr int nDim=TensorTraits::NumDimensions;
    Eigen::Index nx=x.rows();
    Eigen::Index nt=getTensorDimensions(table)[nDim-1];
    Eigen::Index m=getTensorSize(table);
    Eigen::Tensor<Scalar,4,TensorTraits::Layout,typename TensorTraits::Index> store;
    for(int d=nDim-1;d>=0;--d){
        m/=nt;
        Eigen::Tensor<Scalar,3,TensorTraits::Layout,typename TensorTraits::Index> tmp2(nDim,nx,m);
        if(d==nDim-1){
            Eigen::Tensor<Scalar,2,TensorTraits::Layout,typename TensorTraits::Index> first=table.reshape(Eigen::array<Eigen::Index,2>{{m,nt}});
            for(Eigen::Index i=0;i<nx;++i){
                std::pair<Eigen::Index,Scalar> idx=_search1d(point[d],x(i,d));
                for(Eigen::Index j=0;j<m;++j){
                    for(Eigen::Index g=0;g<nDim;++g){
                        if(d==g){
                            tmp2(g,i,j)=(first(j,idx.first+1)-first(j,idx.first))/(point[d](idx.first+1)-point[d](idx.first));
                        }else{
                            tmp2(g,i,j)=idx.second*(first(j,idx.first+1)-first(j,idx.first))+first(j,idx.first);
                        }
                    }
                }
            }
        }else{
            for(Eigen::Index i=0;i<nx;++i){
                std::pair<Eigen::Index,Scalar> idx=_search1d(point[d],x(i,d));
                for(Eigen::Index j=0;j<m;++j){
                    for(Eigen::Index g=0;g<nDim;++g){
                        if(d==g){
                            tmp2(g,i,j)=(store(g,i,j,idx.first+1)-store(g,i,j,idx.first))/(point[d](idx.first+1)-point[d](idx.first));
                        }else{
                            tmp2(g,i,j)=idx.second*(store(g,i,j,idx.first+1)-store(g,i,j,idx.first))+store(g,i,j,idx.first);
                        }
                    }
                }
            }
        }
        nt=(d>=1)?getTensorDimensions(table)[d-1]:1;
        store=tmp2.reshape(Eigen::array<Eigen::Index,4>{{nDim,nx,m/nt,nt}});
    }
    typedef Eigen::Map<Eigen::Matrix<Scalar,-1,-1,TensorTraits::Layout>> maptype;
    return maptype(store.data(),nDim,nx);
}

/**
 * @brief nloptの最適化対象にstd::functionを与えるためのdelegator
 * 
 * @code
 *  // 評価関数をラムダ式等で実装する。
 *  auto obj=[&](unsigned int n,const double* x,double* grad,void* data){
 *      ...
 *      return value_objective_function;
 *  }
 * 
 *  // nlopt_delegatorでラップする
 *  nlopt_delegator obj_d(&obj,nullptr); // 第2引数は上記のラムダに渡すdata
 *  nlopt::opt opt=nlopt::opt(nlopt::algorithm::LN_BOBYQA,6); //アルゴリズムと変数の次元を指定する。
 *  
 *  // nloptの目的関数にはnlopt_delegateeの関数ポインタと、それに渡すdataとしてnlopt_delegatorのポインタを渡す。
 *  opt.set_min_objective(nlopt_delegatee,&obj_d);
 * 
 * @endcode
 */
struct PYBIND11_EXPORT nlopt_delegator{
    std::function<double(unsigned int,const double*,double*,void*)>* func;
    void* data;
    nlopt_delegator(std::function<double(unsigned int,const double*,double*,void*)>* func_,void* data_);
};
/**
 * @brief nloptに最適化対象として渡す関数
 * 
 * @code
 *  // 評価関数をラムダ式等で実装する。
 *  auto obj=[&](unsigned int n,const double* x,double* grad,void* data){
 *      ...
 *      return value_objective_function;
 *  }
 * 
 *  // nlopt_delegatorでラップする
 *  nlopt_delegator obj_d(&obj,nullptr); // 第2引数は上記のラムダに渡すdata
 *  nlopt::opt opt=nlopt::opt(nlopt::algorithm::LN_BOBYQA,6); //アルゴリズムと変数の次元を指定する。
 *  
 *  // nloptの目的関数にはnlopt_delegateeの関数ポインタと、それに渡すdataとしてnlopt_delegatorのポインタを渡す。
 *  opt.set_min_objective(nlopt_delegatee,&obj_d);
 * 
 * @endcode
 */
double PYBIND11_EXPORT nlopt_delegatee(unsigned int n,const double* x,double* grad,void* data);

/**
 * @brief 有本・ポッターの解法でLQRの最適ゲインを求める。
 * 
 * @details 状態方程式が\f$ dz/dt=Az+Bv \f$、評価関数が\f$ J=\int (z^T Q z + v^T R v) dt \f$ である系に対し、
 *      最適な制御入力を \f$ v=-R^{-1} B^T P z \f$ として与える行列 \f$ P \f$ を返す。
 */
Eigen::MatrixXd PYBIND11_EXPORT ArimotoPotter(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);

/**
 * @brief 4段4次のルンゲクッタ法で積分を行う。
 * 
 * @param [in] x0    元の状態量ベクトル
 * @param [in] t0    元の時刻(s)
 * @param [in] dt    進める時間(s)
 * @param [in] deriv x,tにおける状態量の導関数値を計算するを返す関数オブジェクト
 * 
 * @returns t=t0+dtにおける状態量ベクトル
 */
Eigen::VectorXd PYBIND11_EXPORT RK4(const Eigen::VectorXd& x0,const Time& t0,double dt,std::function<Eigen::VectorXd (const Eigen::VectorXd& x,const Time& t)> deriv);

void exportMathUtility(py::module &m);

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
