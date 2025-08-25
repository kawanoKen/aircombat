// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "Common.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include "Utility.h"

namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS(Quaternion)
    //Eigen::Quaternionをラップする
    private:
    Eigen::Quaternion<double,Eigen::DontAlign> q;
    public:
    Quaternion();
    Quaternion(const double &w,const Eigen::Vector3d &vec_);
    Quaternion(const double &w,const double &x,const double &y,const double &z);
    Quaternion(const Eigen::Vector4d &v);
    Quaternion(const Eigen::Quaternion<double> &other);
    Quaternion(const Quaternion& other);
    double& w();
    const double& w() const;
    void w(const double &w_);
    double& x();
    const double& x() const;
    void x(const double &x_);
    double& y();
    const double& y() const;
    void y(const double &y_);
    double& z();
    const double& z() const;
    void z(const double &z_);
    const double& wGetterPY();
    void wSetterPY(const double &w_);
    const double& xGetterPY();
    void xSetterPY(const double &x_);
    const double& yGetterPY();
    void ySetterPY(const double &y_);
    const double& zGetterPY();
    void zSetterPY(const double &z_);
    Eigen::Map<Eigen::Vector3d> vec();
    Eigen::Vector3d vec() const;
    void vec(const Eigen::Vector3d &vec_);
    Eigen::Map<Eigen::Vector3d> vecGetterPY();
    void vecSetterPY(const Eigen::Vector3d &vec_);
    std::string toString() const;
    Eigen::Vector4d toArray() const;
    void fromArray(const Eigen::Vector4d& array);
    double norm() const;
    void normalize();
    Quaternion normalized() const;
    Quaternion operator+(const Quaternion &other) const;
    Quaternion operator-(const Quaternion &other) const;
    Quaternion operator-() const;
    Quaternion operator*(const Quaternion &other) const;
    double dot(const Quaternion &other) const;
    Quaternion conjugate() const;
    Eigen::Vector3d transformVector(const Eigen::Vector3d &v) const;
    Eigen::Matrix3d toRotationMatrix() const;
    Eigen::Vector3d toRotationVector() const;
    Quaternion slerp(const double &t,const Quaternion &other) const;
    Eigen::Matrix<double,3,4> dC1dq() const;
    Eigen::Matrix<double,3,4> dC2dq() const;
    Eigen::Matrix<double,3,4> dC3dq() const;
    Eigen::Matrix<double,3,4> dR1dq() const;
    Eigen::Matrix<double,3,4> dR2dq() const;
    Eigen::Matrix<double,3,4> dR3dq() const;
    Eigen::Matrix<double,4,3> dqdwi() const;
    static Quaternion fromAngle(const Eigen::Vector3d &ax,const double &theta);
    static Quaternion fromBasis(const Eigen::Vector3d &ex,const Eigen::Vector3d &ey,const Eigen::Vector3d &ez);
    static Quaternion fromRotationMatrix(const Eigen::Matrix3d& R);
    template<class Archive>
    void save(Archive & archive) const{
        archive( ::cereal::make_size_tag( static_cast<::cereal::size_type>(4) ));
        for(auto&& value : toArray()){
            archive(value);
        }
    }
    template<class Archive>
    void load(Archive & archive) {
        if constexpr (traits::same_archive_as<Archive,util::NLJSONInputArchive>){
            const nl::json& j=archive.getCurrentNode();
            if(j.is_object()){
                w(j.at("w").get<double>());
                vec(j.at("vec").get<Eigen::Vector3d>());
                return;
            }else if(j.is_array() && j.size()==4){
                //pass
            }else{
                throw std::runtime_error("Quaternion only accepts json array or object.");
            }
        }
        ::cereal::size_type size;
        archive( ::cereal::make_size_tag(size));
        assert(size==4);
        Eigen::Vector4d asArray;
        for(std::size_t i=0;i<4;++i){
            archive(asArray(i));
        }
        fromArray(asArray);
    }
};

void exportQuaternion(py::module &m);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
