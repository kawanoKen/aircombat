// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// numpy.arrayのシリアライゼーション機能を提供する。
//
#pragma once
#include "../../traits/traits.h"
#include "eigen.h"
#include "nljson.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

    inline bool is_row_major(const ::pybind11::array& m){
        return ::pybind11::detail::check_flags(m.ptr(), ::pybind11::array::c_style);
    }
    inline bool is_col_major(const ::pybind11::array& m){
        return ::pybind11::detail::check_flags(m.ptr(), ::pybind11::array::f_style);
    }
    inline bool is_forcecast(const ::pybind11::array& m){
        return ::pybind11::detail::check_flags(m.ptr(), ::pybind11::array::forcecast);
    }
    inline bool is_row_major(int flags){
        return (::pybind11::array::c_style == (flags & ::pybind11::array::c_style));
    }
    inline bool is_col_major(int flags){
        return (::pybind11::array::f_style == (flags & ::pybind11::array::f_style));
    }
    inline bool is_forcecast(int flags){
        return (::pybind11::array::forcecast == (flags & ::pybind11::array::forcecast));
    }

    template<typename Scalar>
    bool check_dtype(const ::pybind11::array& m){
        namespace py=pybind11;
        const auto &api = py::detail::npy_api::get();
        return api.PyArray_EquivTypes_(py::detail::array_proxy(m.ptr())->descr,
                                          py::dtype::of<Scalar>().ptr());
    }

    inline DTypes get_dtype_enum(const ::pybind11::array& m ){
        namespace py=pybind11;

        if(check_dtype<double>(m)){
            return DTypes::DOUBLE_;
        }else if(check_dtype<float>(m)){
            return DTypes::FLOAT_;
        }else if(check_dtype<std::uint64_t>(m)){
            return DTypes::UINT64_;
        }else if(check_dtype<std::int64_t>(m)){
            return DTypes::INT64_;
        }else if(check_dtype<std::uint32_t>(m)){
            return DTypes::UINT32_;
        }else if(check_dtype<std::int32_t>(m)){
            return DTypes::INT32_;
        }else if(check_dtype<std::uint16_t>(m)){
            return DTypes::UINT16_;
        }else if(check_dtype<std::int16_t>(m)){
            return DTypes::INT16_;
        }else if(check_dtype<std::uint8_t>(m)){
            return DTypes::UINT8_;
        }else if(check_dtype<std::int8_t>(m)){
            return DTypes::INT8_;
        }else if(check_dtype<bool>(m)){
            return DTypes::BOOL_;
        }else{
            return DTypes::INVALID_;
            throw cereal::Exception("Unsupported dtype to serialize. dtype="+py::repr(m.dtype()).cast<std::string>());
        }
    }
    inline py::dtype get_dtype(DTypes dtype){
        if(dtype==DTypes::DOUBLE_){
            return py::dtype::of<double>();
        }else if(dtype==DTypes::FLOAT_){
            return py::dtype::of<float>();
        }else if(dtype==DTypes::UINT64_){
            return py::dtype::of<std::uint64_t>();
        }else if(dtype==DTypes::INT64_){
            return py::dtype::of<std::int64_t>();
        }else if(dtype==DTypes::UINT32_){
            return py::dtype::of<std::uint32_t>();
        }else if(dtype==DTypes::INT32_){
            return py::dtype::of<std::int32_t>();
        }else if(dtype==DTypes::UINT16_){
            return py::dtype::of<std::uint16_t>();
        }else if(dtype==DTypes::INT16_){
            return py::dtype::of<std::int16_t>();
        }else if(dtype==DTypes::UINT8_){
            return py::dtype::of<std::uint8_t>();
        }else if(dtype==DTypes::INT8_){
            return py::dtype::of<std::int8_t>();
        }else if(dtype==DTypes::BOOL_){
            return py::dtype::of<bool>();
        }else{
            throw cereal::Exception("Unsupported dtype to serialize.");
        }
    }

    inline pybind11::ssize_t extract_ndim_helper(const nl::ordered_json& j){
        if(j.is_array()){
            if(j.size()>0){
                return 1+extract_ndim_helper(j.at(0));
            }else{
                return 1;
            }
        }else{
            return 0;
        }
    }

    template <typename Scalar, int ExtraFlags>
    struct inner_numpy_array_t_wrapper;

    template <class Archive, typename Scalar, int ExtraFlags>
    inline void save_numpy_array_t_without_swap( Archive & ar, const ::pybind11::array_t<Scalar,ExtraFlags>& m ){
        namespace py=pybind11;
        
        const bool isRowMajorType = is_row_major(m);
        const bool isColMajorType = is_col_major(m);
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;

        py::buffer_info info=m.request(false);
        if(info.ndim==1){
            ar( ::cereal::make_size_tag( (::cereal::size_type)(info.size) ) );
            if constexpr (cereal::traits::is_output_serializable<cereal::BinaryData<const Scalar*>, Archive>::value) {
                if(info.strides[0]==info.itemsize){
                    ar( ::cereal::binary_data( m.data(), info.size * info.itemsize ) );
                    return;
                }
            }
            auto ref=m.template unchecked<1>();
            for(pybind11::ssize_t i=0;i<info.size;++i){
                ar(ref[i]);
            }
        }else{
            // RowMajorは最後の軸に沿って連続、最初の軸が最外周
            // ColMajorは最初の軸に沿って連続、最後の軸が最外周
            const pybind11::ssize_t outerDimIndex = (isRowMajorType ? 0 : info.ndim-1);
            ar( ::cereal::make_size_tag( (::cereal::size_type)(info.shape[outerDimIndex]) ) );
            for(pybind11::ssize_t i=0;i<info.shape[outerDimIndex];++i){
                if(isRowMajorType){
                    auto sub=::pybind11::array_t<Scalar,py::array::c_style>::ensure(m[py::make_tuple(i,py::ellipsis())]);
                    ar(inner_numpy_array_t_wrapper(sub));
                }else{
                    auto sub=::pybind11::array_t<Scalar,py::array::f_style>::ensure(m[py::make_tuple(py::ellipsis(),i)]);
                    ar(inner_numpy_array_t_wrapper(sub));
                }
            }
        }
    }

    template <class Archive, typename Scalar, int ExtraFlags>
    inline void load_numpy_array_t_without_swap( Archive & ar, ::pybind11::array_t<Scalar,ExtraFlags> & m , bool noResize){
        namespace py=pybind11;
        
        const bool isRowMajorType = is_row_major(ExtraFlags);
        const bool isColMajorType = is_col_major(ExtraFlags);
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;

        if(!m.writeable()){
            throw cereal::Exception("Destination is not a writeable py::array.");
        }

        py::buffer_info info=m.request(true);
        if(info.ndim==1){
            cereal::size_type size;
            ar( cereal::make_size_tag( size ) );
            if(m.owndata() && !noResize){
                try{
                    m.resize({(pybind11::ssize_t)size},false);
                }catch(std::exception& ex){
                    assert(info.size==size);
                }
            }else{
                assert(info.size==size);
            }
            if constexpr ( cereal::traits::is_input_serializable<cereal::BinaryData<Scalar*>, Archive>::value ) {
                if(info.strides[0]==info.itemsize){
                    ar( ::cereal::binary_data( m.mutable_data(), size * info.itemsize ) );
                    return;
                }
            }
            auto ref=m.template mutable_unchecked<1>();
            for(pybind11::ssize_t i=0;i<size;++i){
                ar(ref[i]);
            }
        }else{
            // RowMajorは最後の軸に沿って連続、最初の軸が最外周
            // ColMajorは最初の軸に沿って連続、最後の軸が最外周
            const pybind11::ssize_t outerDimIndex = (isRowMajorType ? 0 : info.ndim-1);

            cereal::size_type outerSize;
            ar( ::cereal::make_size_tag( outerSize ) );
            for(pybind11::ssize_t i=0; i<outerSize; ++i){
                if(i==0){
                    ::pybind11::array_t<Scalar,ExtraFlags> sub(std::vector<pybind11::ssize_t>(info.ndim-1,1));
                    auto wrapped=inner_numpy_array_t_wrapper(sub,false);
                    ar(wrapped);
                    if(m.owndata() && !noResize){
                        std::vector<pybind11::ssize_t> newShape(sub.request(true).shape);
                        if(isRowMajorType){
                            newShape.insert(newShape.begin(),outerSize);
                        }else{
                            newShape.push_back(outerSize);
                        }
                        try{
                            // resizeはstorage orderが無視されるため新たなarrayの代入とする。
                            m=::pybind11::array_t<Scalar,ExtraFlags>(newShape);
                        }catch(...){
                            assert(info.shape[outerDimIndex]==outerSize);
                        }
                    }else{
                        assert(info.shape[outerDimIndex]==outerSize);
                    }
                    if(isRowMajorType){
                        m[py::make_tuple(i,py::ellipsis())]=sub;
                    }else{
                        m[py::make_tuple(py::ellipsis(),i)]=sub;
                    }
                }else{
                    if(isRowMajorType){
                        ::pybind11::array_t<Scalar,py::array::c_style> sub(m[py::make_tuple(i,py::ellipsis())]);
                        auto wrapped=inner_numpy_array_t_wrapper(sub,true);
                        ar(wrapped);
                    }else{
                        ::pybind11::array_t<Scalar,py::array::f_style> sub(m[py::make_tuple(py::ellipsis(),i)]);
                        auto wrapped=inner_numpy_array_t_wrapper(sub,true);
                        ar(wrapped);
                    }
                }
            }
        }
    }

    template <typename Scalar, int ExtraFlags>
    struct inner_numpy_array_t_wrapper{
        inner_numpy_array_t_wrapper(::pybind11::array_t<Scalar,ExtraFlags>& m_,bool noResize_=true):m(m_),noResize(noResize_){}
        template <class Archive>
        inline void save(Archive & ar) const{
            save_numpy_array_t_without_swap(ar,m);
        }
        template <class Archive>
        inline void load(Archive & ar){
            load_numpy_array_t_without_swap(ar,m,noResize);
        }
        ::pybind11::array_t<Scalar,ExtraFlags>& m;
        bool noResize;
    };
    template <class Archive, typename Scalar, int ExtraFlags>
    inline void save_numpy_array_t( Archive & ar, const ::pybind11::array_t<Scalar,ExtraFlags>& m )
    {
        namespace py=pybind11;
        
        const bool isRowMajorType = is_row_major(m);
        const bool isColMajorType = is_col_major(m);
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;

        py::buffer_info info=m.request(false);

        constexpr bool always_rowmajor = is_text_archive && serialize_eigen_as_text_always_rowmajor;
        constexpr bool save_ndim = !traits::same_archive_as<Archive,::asrc::core::util::NLJSONOutputArchive>;
        constexpr bool use_nvp = (!always_rowmajor || save_ndim);
        
        if constexpr (save_ndim){
            // NLJSONArchive以外はdtypeとndimが既知でないと読み込めないので出力しておく必要あり
            save_dtype_helper<Scalar>(ar);
            ar(::cereal::make_nvp("ndim",info.ndim));
        }

        if constexpr (!always_rowmajor){
            // つねにRowMajorを強制するのでなければLayOut情報の出力が必要
            ar(::cereal::make_nvp("isRowMajorSrc",isRowMajorType));            
        }

        if constexpr (always_rowmajor){
            if(!isRowMajorType){
                // ColMajorをRowMajorとして出力
                // おそらくswapしたtensorを複製した方が速い
                auto swapped=py::array_t<Scalar,py::array::c_style>::ensure(m);
                if constexpr (use_nvp){
                    ar(::cereal::make_nvp("value",inner_numpy_array_t_wrapper(swapped)));
                }else{
                    save_numpy_array_t_without_swap(ar,swapped);
                }
                return;
            }
        }
        // as-is
        if constexpr (use_nvp){
            auto non_const=py::reinterpret_borrow<py::array_t<Scalar,ExtraFlags>>(m);
            ar(::cereal::make_nvp("value",inner_numpy_array_t_wrapper(non_const)));
        }else{
            save_numpy_array_t_without_swap(ar,m);
        }
    }

    template <class Archive, typename Scalar, int ExtraFlags>
    inline void load_numpy_array_t( Archive & ar, ::pybind11::array_t<Scalar,ExtraFlags> & m )
    {
        namespace py=pybind11;
        
        const bool isRowMajorType = is_row_major(m);
        const bool isColMajorType = is_col_major(m);
        const bool isForcecast = is_forcecast(m);
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;

        if(!m.writeable()){
            throw cereal::Exception("Destination is not a writeable py::array.");
        }

        constexpr bool always_rowmajor = is_text_archive && serialize_eigen_as_text_always_rowmajor;
        constexpr bool save_ndim = !traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>;
        constexpr bool use_nvp = (!always_rowmajor || save_ndim);

        pybind11::ssize_t srcNdim;
        if constexpr (save_ndim){
            // NLJSONArchive以外は階数(ndim)をArchiveから抽出
            ar(::cereal::make_nvp("ndim",srcNdim));
        }else{
            // NLJSONArchiveはjsonの形から推論
            const nl::ordered_json& j=ar.getCurrentNode();
            assert(j.is_array() && j.size()>0);

            srcNdim=extract_ndim_helper(j);
        }

        bool isRowMajorSrc;
        if constexpr (!always_rowmajor){
            // つねにRowMajorを強制するのでなければLayOutを抽出
            ar(::cereal::make_nvp("isRowMajorSrc",isRowMajorSrc));
        }else{
            isRowMajorSrc=true;
        }

        py::buffer_info info=m.request(true);

        if(isForcecast || isRowMajorType!=isRowMajorSrc || (srcNdim>1 && isColMajorType!=!isRowMajorSrc) || info.ndim!=srcNdim){
            // forcecastの場合又はLayoutか階数が異なる場合はそれらを合わせた新規オブジェクトに読み込んでから代入
            if(isRowMajorSrc){
                py::array_t<Scalar,py::array::c_style> tmp{std::vector<pybind11::ssize_t>(srcNdim,1)};
                if constexpr (use_nvp){
                    ar(::cereal::make_nvp("value",inner_numpy_array_t_wrapper(tmp,false)));
                }else{
                    load_numpy_array_t_without_swap(ar,tmp,false);
                }
                m=tmp;
            }else{
                py::array_t<Scalar,py::array::f_style> tmp{std::vector<pybind11::ssize_t>(srcNdim,1)};
                if constexpr (use_nvp){
                    ar(::cereal::make_nvp("value",inner_numpy_array_t_wrapper(tmp,false)));
                }else{
                    load_numpy_array_t_without_swap(ar,tmp,false);
                }
                m=tmp;
            }
        }else{
            // as-is
            if constexpr (use_nvp){
                ar(::cereal::make_nvp("value",inner_numpy_array_t_wrapper(m,false)));
            }else{
                load_numpy_array_t_without_swap(ar,m,false);
            }
        }
    }

    template <class Archive,typename Scalar>
    inline void save_numpy_array_as_dtype_of(Archive & ar, const ::pybind11::array& m){
        namespace py=pybind11;

        const bool isRowMajorType = is_row_major(m);
        const bool isColMajorType = is_col_major(m);
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;

        if(isColMajorType){
            auto casted=py::array_t<Scalar,py::array::f_style>::ensure(m);
            save_numpy_array_t(ar,casted);
        }else{// if(isRowMajorType){
            auto casted=py::array_t<Scalar,py::array::c_style>::ensure(m);
            save_numpy_array_t(ar,casted);
        }
    }

    template <class Archive,typename Scalar>
    inline void load_numpy_array_as_dtype_of(Archive & ar, ::pybind11::array& m, int ExtraFlags){
        namespace py=pybind11;

        const bool isRowMajorType = is_row_major(ExtraFlags);
        const bool isColMajorType = is_col_major(ExtraFlags);
        const bool isForcecast = is_forcecast(ExtraFlags);
        if(isForcecast){
            auto casted=py::array_t<Scalar,py::array::forcecast>::ensure(m);
            if(casted){
                load_numpy_array_t(ar,casted);
                m=casted;
            }else{
                py::array_t<Scalar,py::array::forcecast> tmp;
                load_numpy_array_t(ar,tmp);
                m=tmp;
            }
        }else if(isColMajorType){
            auto casted=py::array_t<Scalar,py::array::f_style>::ensure(m);
            if(casted){
                load_numpy_array_t(ar,casted);
                m=casted;
            }else{
                py::array_t<Scalar,py::array::f_style> tmp;
                load_numpy_array_t(ar,tmp);
                m=tmp;
            }
        }else{// if(isRowMajorType){
            auto casted=py::array_t<Scalar,py::array::c_style>::ensure(m);
            if(casted){
                load_numpy_array_t(ar,casted);
                m=casted;
            }else{
                py::array_t<Scalar,py::array::c_style> tmp;
                load_numpy_array_t(ar,tmp);
                m=tmp;
            }
        }
    }

    template <class Archive>
    inline void save_numpy_array( Archive & ar, const ::pybind11::array& m ){
        namespace py=pybind11;

        if(check_dtype<double>(m)){
            save_numpy_array_as_dtype_of<Archive,double>(ar,m);
        }else if(check_dtype<float>(m)){
            save_numpy_array_as_dtype_of<Archive,float>(ar,m);
        }else if(check_dtype<std::uint64_t>(m)){
            save_numpy_array_as_dtype_of<Archive,std::uint64_t>(ar,m);
        }else if(check_dtype<std::int64_t>(m)){
            save_numpy_array_as_dtype_of<Archive,std::int64_t>(ar,m);
        }else if(check_dtype<std::uint32_t>(m)){
            save_numpy_array_as_dtype_of<Archive,std::uint32_t>(ar,m);
        }else if(check_dtype<std::int32_t>(m)){
            save_numpy_array_as_dtype_of<Archive,std::int32_t>(ar,m);
        }else if(check_dtype<std::uint16_t>(m)){
            save_numpy_array_as_dtype_of<Archive,std::uint16_t>(ar,m);
        }else if(check_dtype<std::int16_t>(m)){
            save_numpy_array_as_dtype_of<Archive,std::int16_t>(ar,m);
        }else if(check_dtype<std::uint8_t>(m)){
            save_numpy_array_as_dtype_of<Archive,std::uint8_t>(ar,m);
        }else if(check_dtype<std::int8_t>(m)){
            save_numpy_array_as_dtype_of<Archive,std::int8_t>(ar,m);
        }else if(check_dtype<bool>(m)){
            save_numpy_array_as_dtype_of<Archive,bool>(ar,m);
        }else{
            throw cereal::Exception("Unsupported dtype to serialize. dtype="+py::repr(m.dtype()).cast<std::string>());
        }
    }


    template <class Archive>
    inline void load_numpy_array( Archive & ar, ::pybind11::array& m , DTypes dtype, int ExtraFlags){
        if(!m.writeable()){
            throw cereal::Exception("Destination is not a writeable py::array.");
        }

        if(dtype==DTypes::DOUBLE_){
            load_numpy_array_as_dtype_of<Archive,double>(ar,m,ExtraFlags);
        }else if(dtype==DTypes::FLOAT_){
            load_numpy_array_as_dtype_of<Archive,float>(ar,m,ExtraFlags);
        }else if(dtype==DTypes::UINT64_){
            load_numpy_array_as_dtype_of<Archive,std::uint64_t>(ar,m,ExtraFlags);
        }else if(dtype==DTypes::INT64_){
            load_numpy_array_as_dtype_of<Archive,std::int64_t>(ar,m,ExtraFlags);
        }else if(dtype==DTypes::UINT32_){
            load_numpy_array_as_dtype_of<Archive,std::uint32_t>(ar,m,ExtraFlags);
        }else if(dtype==DTypes::INT32_){
            load_numpy_array_as_dtype_of<Archive,std::int32_t>(ar,m,ExtraFlags);
        }else if(dtype==DTypes::UINT16_){
            load_numpy_array_as_dtype_of<Archive,std::uint16_t>(ar,m,ExtraFlags);
        }else if(dtype==DTypes::INT16_){
            load_numpy_array_as_dtype_of<Archive,std::int16_t>(ar,m,ExtraFlags);
        }else if(dtype==DTypes::UINT8_){
            load_numpy_array_as_dtype_of<Archive,std::uint8_t>(ar,m,ExtraFlags);
        }else if(dtype==DTypes::INT8_){
            load_numpy_array_as_dtype_of<Archive,std::int8_t>(ar,m,ExtraFlags);
        }else if(dtype==DTypes::BOOL_){
            load_numpy_array_as_dtype_of<Archive,bool>(ar,m,ExtraFlags);
        }else{
            throw cereal::Exception("Unsupported dtype to serialize. dtype="+py::repr(m.dtype()).cast<std::string>());
        }
    }

    struct numpy_array_wrapper{
        numpy_array_wrapper(::pybind11::array& m_):m(m_),withHint(false){}
        numpy_array_wrapper(::pybind11::array& m_,DTypes dtype_,bool is_col_major_):m(m_),withHint(true),dtype(dtype_),is_col_major(is_col_major_){}
        template <class Archive>
        inline void save(Archive & ar) const{
            save_numpy_array(ar,m);
        }
        template <class Archive>
        inline void load(Archive & ar){
            int ExtraFlags;
            if(withHint){
                if constexpr (!traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>){
                    auto srcDtype=load_dtype_helper(ar);
                    assert(srcDtype==dtype);
                }
                if(get_dtype_enum(m)!=dtype){
                    m=py::array(get_dtype(dtype),0);
                }
                ExtraFlags = is_col_major ? py::array::f_style : py::array::c_style;
            }else{
                dtype=load_dtype_helper(ar);
                ExtraFlags = py::array::forcecast;
                m=py::array(get_dtype(dtype),0);
            }
            load_numpy_array(ar,m,dtype,ExtraFlags);
        }
        ::pybind11::array& m;
        bool withHint;
        DTypes dtype;
        bool is_col_major;
    };

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
