// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// Eigen::Matrix, Eigen::Tensorをcerealシリアライズする機能を提供する。
//
#pragma once
#include "../../traits/traits.h"
#include "nljson.h"

#include <Eigen/Core>
#include <Eigen/CXX11/Tensor>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>

#ifndef ASRC_SERIALIZE_EIGEN_AS_TEXT_ALWAYS_ROWMAJOR
    #define ASRC_SERIALIZE_EIGEN_AS_TEXT_ALWAYS_ROWMAJOR true
#endif

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

    constexpr bool serialize_eigen_as_text_always_rowmajor = ASRC_SERIALIZE_EIGEN_AS_TEXT_ALWAYS_ROWMAJOR;

    template< class EigenType>
    struct inner_eigen_type_wrapper; // forward declaration

    enum class DTypes : std::uint8_t {
        INVALID_=0,
        DOUBLE_,
        FLOAT_,
        UINT64_,
        INT64_,
        UINT32_,
        INT32_,
        UINT16_,
        INT16_,
        UINT8_,
        INT8_,
        BOOL_
    };

    template <class Scalar,class Archive>
    inline void save_dtype_helper(Archive& ar){
        if constexpr (
            !traits::same_archive_as<Archive,::asrc::core::util::NLJSONOutputArchive>
        ){
            DTypes dtype;
            if constexpr (std::same_as<Scalar,double>){
                dtype=DTypes::DOUBLE_;
            }else if constexpr (std::same_as<Scalar,float>){
                dtype=DTypes::FLOAT_;
            }else if constexpr (std::same_as<Scalar,std::uint64_t>){
                dtype=DTypes::UINT64_;
            }else if constexpr (std::same_as<Scalar,std::int64_t>){
                dtype=DTypes::INT64_;
            }else if constexpr (std::same_as<Scalar,std::uint32_t>){
                dtype=DTypes::UINT32_;
            }else if constexpr (std::same_as<Scalar,std::int32_t>){
                dtype=DTypes::INT32_;
            }else if constexpr (std::same_as<Scalar,std::uint16_t>){
                dtype=DTypes::UINT16_;
            }else if constexpr (std::same_as<Scalar,std::int16_t>){
                dtype=DTypes::INT16_;
            }else if constexpr (std::same_as<Scalar,std::uint8_t>){
                dtype=DTypes::UINT8_;
            }else if constexpr (std::same_as<Scalar,std::int8_t>){
                dtype=DTypes::INT8_;
            }else if constexpr (std::same_as<Scalar,bool>){
                dtype=DTypes::BOOL_;
            }
            ar(::cereal::make_nvp("dtype",dtype));
        }
    }

    inline DTypes extract_dtype_helper(const nl::ordered_json& j){
        // 一つでも実数値があったらdouble、そうでなければsigned int64とする。
        if(j.at(0).is_array()){
            for(auto && v:j){
                DTypes ret=extract_dtype_helper(v);
                if(ret==DTypes::DOUBLE_){
                    return ret;
                }
            }
            return DTypes::INT64_;
        }else if(j.at(0).is_number()){
            for(auto && v:j){
                if(v.is_number_float()){
                return DTypes::DOUBLE_;
                }
            }
            return DTypes::INT64_;
        }else{
            throw cereal::Exception("json to be loaded as py::array must be an array of number.");
        }
    }

    template <class Archive>
    inline DTypes load_dtype_helper(Archive& ar){
        if constexpr (
            traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>
        ){
            const nl::ordered_json& j=ar.getCurrentNode();
            assert(j.is_array() && j.size()>0);
            return extract_dtype_helper(j);
        }else{
            DTypes dtype;
            ar(::cereal::make_nvp("dtype",dtype));
            return dtype;
        }
    }

    //
    // Eigen::DenseBase
    //
    template <class Archive, traits::matrix MatrixType>
    inline void save_without_swap( Archive & ar, MatrixType const & m )
    {
        using MatrixTraits=Eigen::internal::traits<MatrixType>;
        constexpr bool isRowMajorType = MatrixType::IsRowMajor;
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;
        
        if constexpr (MatrixTraits::RowsAtCompileTime==1 || MatrixTraits::ColsAtCompileTime==1) {
            ar( ::cereal::make_size_tag( static_cast<::cereal::size_type>(m.size()) ) );
            if constexpr (
                cereal::traits::is_output_serializable<cereal::BinaryData<const typename MatrixTraits::Scalar *>, Archive>::value
                && (
                    std::derived_from<MatrixType,Eigen::PlainObjectBase<MatrixType>>
                    || std::derived_from<MatrixType,Eigen::MapBase<MatrixType>>
                )
            ) {
                // Eigen::PlainObjectBase (Eigen::Matrix, Eigen::Vector, Eigen::Array等)
                // Eigen::MapBase (Eigen::Map, Eigen::Ref, Eigen::Block等)
                // m.data()でデータにアクセス可能なクラス全般
                ar( ::cereal::binary_data( m.data(), m.size() * sizeof(typename MatrixTraits::Scalar) ) );
            }else{
                for(auto&& v : m){
                    ar(v);
                }
            }
        }else{
            ar( ::cereal::make_size_tag( static_cast<::cereal::size_type>(m.outerSize()) ) );
            for(Eigen::Index outer=0; outer<m.outerSize(); ++outer){
                // m.innerVector(outer)を使いたいところだが、
                // MatrixTraits::ColsAtCompileTime==Eigen::Dynamicを要求されてしまうようなので、
                // 固定サイズの行列には使用できない。
                if constexpr (isRowMajorType) {
                    ar(m.row(outer));
                }else{
                    ar(m.col(outer));
                }
            }
        }
    }
    //
    // loadはnon-constなPlainObjectBase又はMapBaseに限る。
    //
    template <class Archive, traits::matrix MatrixType>
    requires (
        (
            std::derived_from<MatrixType,Eigen::PlainObjectBase<MatrixType>>
            || std::derived_from<MatrixType,Eigen::MapBase<MatrixType>>
        )
        && !std::is_const_v<MatrixType>
    )
    inline void load_without_swap( Archive & ar, MatrixType & m )
    {
        using MatrixTraits=Eigen::internal::traits<MatrixType>;
        constexpr bool isRowMajorType = MatrixType::IsRowMajor;
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;
        
        if constexpr (MatrixTraits::RowsAtCompileTime==1 || MatrixTraits::ColsAtCompileTime==1) {
            cereal::size_type size;
            ar( cereal::make_size_tag( size ) );
            if constexpr (std::derived_from<MatrixType,Eigen::PlainObjectBase<MatrixType>>){
                m.resize(size);
            }else{
                assert(m.size()==size);
            }
            if constexpr (cereal::traits::is_input_serializable<cereal::BinaryData<typename MatrixTraits::Scalar *>, Archive>::value) {
                ar( cereal::binary_data( m.data(), static_cast<std::size_t>( size ) * sizeof(typename MatrixTraits::Scalar) ) );
            }else{
                for(Eigen::Index i=0;i<size;++i){
                    ar(m(i));
                }
            }
        }else{
            cereal::size_type outerSize;
            ar( cereal::make_size_tag( outerSize ) );
            for(Eigen::Index outer=0; outer<outerSize; ++outer){
                if(outer==0){
                    if constexpr (isRowMajorType) {
                        Eigen::Matrix<
                            typename MatrixTraits::Scalar,
                            1,
                            MatrixTraits::ColsAtCompileTime,
                            Eigen::RowMajor,
                            1,
                            MatrixTraits::MaxColsAtCompileTime
                        > tmp;
                        ar(tmp);
                        if constexpr (std::derived_from<MatrixType,Eigen::PlainObjectBase<MatrixType>>){
                            m.resize(outerSize,tmp.cols());
                        }else{
                            assert(tmp.cols()==m.innerSize() && m.outerSize()==outerSize);
                        }
                        // m.innerVector(outer)を使いたいところだが、
                        // MatrixTraits::ColsAtCompileTime==Eigen::Dynamicを要求されてしまうようなので、
                        // 固定サイズの行列には使用できない。
                        m.row(outer)=std::move(tmp);
                    }else{
                        Eigen::Matrix<
                            typename MatrixTraits::Scalar,
                            MatrixTraits::RowsAtCompileTime,
                            1,
                            Eigen::ColMajor,
                            MatrixTraits::MaxRowsAtCompileTime,
                            1
                        > tmp;
                        ar(tmp);
                        if constexpr (std::derived_from<MatrixType,Eigen::PlainObjectBase<MatrixType>>){
                            m.resize(tmp.rows(),outerSize);
                        }else{
                            assert(tmp.rows()==m.innerSize() && m.outerSize()==outerSize);
                        }
                        m.col(outer)=std::move(tmp);
                    }
                }else{
                    if constexpr (isRowMajorType) {
                        ar(m.row(outer));
                    }else{
                        ar(m.col(outer));
                    }
                }
            }
        }
    }

    //
    // Eigen::Tensor
    //
    template<class Archive, typename Scalar_, int NumIndices_, int Options_, typename IndexType_>
    inline void save_without_swap( Archive & ar, Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType_> const & m )
    {
        using Type = Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType_>;
        using traits=Eigen::internal::traits<Type>;
        constexpr bool isRowMajorType = static_cast<int>(traits::Layout) == static_cast<int>(Eigen::RowMajor);
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;

        if constexpr (traits::NumDimensions==1) {
            ar( ::cereal::make_size_tag( static_cast<::cereal::size_type>(m.size()) ) );
            if constexpr (cereal::traits::is_output_serializable<cereal::BinaryData<const typename traits::Scalar *>, Archive>::value) {
                ar( ::cereal::binary_data( m.data(), m.size() * sizeof(typename traits::Scalar) ) );
            }else{
                for(Eigen::Index i=0; i<m.size(); ++i){
                    ar(m(i));
                }
            }
        }else{
            // RowMajorは最後の軸に沿って連続、最初の軸が最外周
            // ColMajorは最初の軸に沿って連続、最後の軸が最外周
            constexpr Eigen::Index outerDimIndex = (isRowMajorType ? 0 : traits::NumDimensions-1);
            constexpr Eigen::Index dimIndexOffset = (isRowMajorType ? 1 : 0);

            ar( ::cereal::make_size_tag( static_cast<::cereal::size_type>(m.dimensions()[outerDimIndex]) ) );
            Eigen::DSizes<Eigen::Index, traits::NumDimensions-1> newDimensions;
            for(Eigen::Index i=0; i<traits::NumDimensions-1; ++i){
                newDimensions[i]=m.dimensions()[i+dimIndexOffset];
            }
            Eigen::Index stride=m.size()/m.dimensions()[outerDimIndex];
            for(Eigen::Index i=0; i<m.dimensions()[outerDimIndex]; ++i){
                Eigen::TensorMap<const Eigen::Tensor<typename traits::Scalar, traits::NumDimensions-1, traits::Options, typename traits::Index>,Options_> mapped(
                    m.data() + stride*i,
                    newDimensions
                );
                ar(mapped);
            }
        }
    }
    template<class Archive, typename Scalar_, int NumIndices_, int Options_, typename IndexType_>
    inline void load_without_swap( Archive & ar, Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType_> & m )
    {
        using Type = Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType_>;
        using traits=Eigen::internal::traits<Type>;
        constexpr bool isRowMajorType = static_cast<int>(traits::Layout) == static_cast<int>(Eigen::RowMajor);
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;

        if constexpr (traits::NumDimensions==1) {
            cereal::size_type size;
            ar( cereal::make_size_tag( size ) );
            m.resize(size);
            if constexpr (cereal::traits::is_input_serializable<cereal::BinaryData<typename traits::Scalar *>, Archive>::value) {
                ar( cereal::binary_data( m.data(), static_cast<std::size_t>( size ) * sizeof(typename traits::Scalar) ) );
            }else{
                for(Eigen::Index i=0;i<size;++i){
                    ar(m(i));
                }
            }
        }else{
            // RowMajorは最後の軸に沿って連続、最初の軸が最外周
            // ColMajorは最初の軸に沿って連続、最後の軸が最外周
            constexpr Eigen::Index outerDimIndex = (isRowMajorType ? 0 : traits::NumDimensions-1);
            constexpr Eigen::Index dimIndexOffset = (isRowMajorType ? 1 : 0);

            cereal::size_type outerSize;
            ar( ::cereal::make_size_tag( outerSize ) );
            Eigen::DSizes<Eigen::Index, traits::NumDimensions> newDimensions;
            Eigen::DSizes<Eigen::Index, traits::NumDimensions-1> subDimensions;
            Eigen::DSizes<Eigen::Index, traits::NumDimensions> startIndices;
            Eigen::DSizes<Eigen::Index, traits::NumDimensions> sliceSizes;
            Eigen::Index stride;
            for(Eigen::Index i=0; i<outerSize; ++i){
                if(i==0){
                    Eigen::Tensor<typename traits::Scalar, traits::NumDimensions-1, traits::Options, typename traits::Index> sub;
                    ar(sub);
                    subDimensions=sub.dimensions();
                    for(Eigen::Index i=0; i<traits::NumDimensions-1; ++i){
                        newDimensions[i+dimIndexOffset]=subDimensions[i];
                        startIndices[i+dimIndexOffset]=0;
                        sliceSizes[i+dimIndexOffset]=subDimensions[i];
                    }
                    newDimensions[outerDimIndex]=outerSize;
                    m.resize(newDimensions);
                    startIndices[outerDimIndex]=0;
                    sliceSizes[outerDimIndex]=1;
                    m.slice(startIndices,sliceSizes)=std::move(sub.reshape(sliceSizes));
                    stride=m.size()/outerSize;
                }else{
                    Eigen::TensorMap<Eigen::Tensor<typename traits::Scalar, traits::NumDimensions-1, traits::Options, typename traits::Index>,Options_> mapped(
                        m.data() + stride*i,
                        subDimensions
                    );
                    ar(mapped);
                }
            }
        }
    }

    //
    // Eigen::TensorMap
    //
    template<class Archive, typename PlainObjectType, int Options_, template <class> class MakePointer_>
    inline void save_without_swap( Archive & ar, Eigen::TensorMap<PlainObjectType, Options_, MakePointer_> const & m )
    {
        using Type = Eigen::TensorMap<PlainObjectType, Options_, MakePointer_>;
        using traits=Eigen::internal::traits<Type>;
        constexpr bool isRowMajorType = static_cast<int>(traits::Layout) == static_cast<int>(Eigen::RowMajor);
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;

        if constexpr (traits::NumDimensions==1) {
            ar( ::cereal::make_size_tag( static_cast<::cereal::size_type>(m.size()) ) );
            if constexpr (cereal::traits::is_output_serializable<cereal::BinaryData<const typename traits::Scalar *>, Archive>::value) {
                ar( ::cereal::binary_data( m.data(), m.size() * sizeof(typename traits::Scalar) ) );
            }else{
                for(Eigen::Index i=0;i<m.size();++i){
                    ar(m(i));
                }
            }
        }else{
            // RowMajorは最後の軸に沿って連続、最初の軸が最外周
            // ColMajorは最初の軸に沿って連続、最後の軸が最外周
            constexpr Eigen::Index outerDimIndex = (isRowMajorType ? 0 : traits::NumDimensions-1);
            constexpr Eigen::Index dimIndexOffset = (isRowMajorType ? 1 : 0);

            ar( ::cereal::make_size_tag( static_cast<::cereal::size_type>(m.dimensions()[outerDimIndex]) ) );
            Eigen::DSizes<Eigen::Index, traits::NumDimensions-1> newDimensions;
            for(Eigen::Index i=0; i<traits::NumDimensions-1; ++i){
                newDimensions[i]=m.dimensions()[i+dimIndexOffset];
            }
            Eigen::Index stride=m.size()/m.dimensions()[outerDimIndex];
            for(Eigen::Index i=0; i<m.dimensions()[outerDimIndex]; ++i){
                Eigen::TensorMap<const Eigen::Tensor<typename traits::Scalar, traits::NumDimensions-1, traits::Options, typename traits::Index>,Options_> mapped(
                    m.data() + stride*i,
                    newDimensions
                );
                ar(mapped);
            }
        }
    }
    template<class Archive, typename PlainObjectType, int Options_, template <class> class MakePointer_>
    inline void load_without_swap( Archive & ar, Eigen::TensorMap<PlainObjectType, Options_, MakePointer_> & m )
    {
        using Type = Eigen::TensorMap<PlainObjectType, Options_, MakePointer_>;
        using traits=Eigen::internal::traits<Type>;
        constexpr bool isRowMajorType = static_cast<int>(traits::Layout) == static_cast<int>(Eigen::RowMajor);
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;

        if constexpr (traits::NumDimensions==1) {
            cereal::size_type size;
            ar( cereal::make_size_tag( size ) );
            assert(m.size()==size);
            if constexpr (cereal::traits::is_input_serializable<cereal::BinaryData<typename traits::Scalar *>, Archive>::value) {
                ar( cereal::binary_data( m.data(), static_cast<std::size_t>( size ) * sizeof(typename traits::Scalar) ) );
            }else{
                for(Eigen::Index i=0;i<size;++i){
                    ar(m(i));
                }
            }
        }else{
            // RowMajorは最後の軸に沿って連続、最初の軸が最外周
            // ColMajorは最初の軸に沿って連続、最後の軸が最外周
            constexpr Eigen::Index outerDimIndex = (isRowMajorType ? 0 : traits::NumDimensions-1);
            constexpr Eigen::Index dimIndexOffset = (isRowMajorType ? 1 : 0);

            cereal::size_type outerSize;
            ar( ::cereal::make_size_tag( outerSize ) );
            assert(m.dimensions()[outerDimIndex]==outerSize);
            Eigen::DSizes<Eigen::Index, traits::NumDimensions> newDimensions;
            Eigen::DSizes<Eigen::Index, traits::NumDimensions-1> subDimensions;
            Eigen::DSizes<Eigen::Index, traits::NumDimensions> startIndices;
            Eigen::DSizes<Eigen::Index, traits::NumDimensions> sliceSizes;
            Eigen::Index stride=m.size()/outerSize;
            for(Eigen::Index i=0; i<traits::NumDimensions-1; ++i){
                subDimensions[i]=m.dimensions()[i+dimIndexOffset];
            }
            for(Eigen::Index i=0; i<outerSize; ++i){
                Eigen::TensorMap<Eigen::Tensor<typename traits::Scalar, traits::NumDimensions-1, traits::Options, typename traits::Index>,Options_> mapped(
                    m.data() + stride*i,
                    subDimensions
                );
                ar(mapped);
            }
        }
    }

    //
    // Eigen::TensorSlicingOp
    //
    template<class Archive, typename StartIndices, typename Sizes, typename XprType>
    inline void save_without_swap( Archive & ar, Eigen::TensorSlicingOp<StartIndices, Sizes, XprType> const & m )
    {
        using Type = Eigen::TensorSlicingOp<StartIndices, Sizes, XprType>;
        using traits=Eigen::internal::traits<Type>;
        constexpr bool isRowMajorType = static_cast<int>(traits::Layout) == static_cast<int>(Eigen::RowMajor);
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;

        if constexpr (Type::NumDimensions==1) {
            ar( ::cereal::make_size_tag( static_cast<::cereal::size_type>(m.size()) ) );
            auto evaluated=Eigen::TensorEvaluator(m,Eigen::DefaultDevice());
            for(Eigen::Index i=0;i<m.size();++i){
                ar(evaluated.coeff(i));
            }
        }else{
            // RowMajorは最後の軸に沿って連続、最初の軸が最外周
            // ColMajorは最初の軸に沿って連続、最後の軸が最外周
            constexpr Eigen::Index outerDimIndex = (isRowMajorType ? 0 : traits::NumDimensions-1);
            constexpr Eigen::Index dimIndexOffset = (isRowMajorType ? 1 : 0);

            ar( ::cereal::make_size_tag( static_cast<::cereal::size_type>(m.sizes()[outerDimIndex]) ) );
            Eigen::DSizes<Eigen::Index, traits::NumDimensions-1> newDimensions;
            Eigen::DSizes<Eigen::Index, traits::NumDimensions> startIndices;
            Eigen::DSizes<Eigen::Index, traits::NumDimensions> sliceSizes=m.sizes();
            for(Eigen::Index i=0; i<traits::NumDimensions-1; ++i){
                newDimensions[i]=m.sizes()[i+dimIndexOffset];
                startIndices[i+dimIndexOffset]=0;
            }
            sliceSizes[outerDimIndex]=1;
            for(Eigen::Index i=0; i<m.sizes()[outerDimIndex]; ++i){
                startIndices[outerDimIndex]=i;
                ar(m.slice(startIndices,sliceSizes).reshape(newDimensions));
            }
        }
    }
    template<class Archive, typename StartIndices, typename Sizes, typename XprType>
    inline void load_without_swap( Archive & ar, Eigen::TensorSlicingOp<StartIndices, Sizes, XprType> & m )
    {
        using Type = Eigen::TensorSlicingOp<StartIndices, Sizes, XprType>;
        using traits=Eigen::internal::traits<Type>;
        constexpr bool isRowMajorType = static_cast<int>(traits::Layout) == static_cast<int>(Eigen::RowMajor);
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;

        if constexpr (Type::NumDimensions==1) {
            cereal::size_type size;
            ar( cereal::make_size_tag( size ) );
            assert(size==m.size());
            auto evaluated=Eigen::TensorEvaluator(m,Eigen::DefaultDevice());
            for(Eigen::Index i=0;i<size;++i){
                ar(evaluated.coeffRef(i));
            }
        }else{
            // RowMajorは最後の軸に沿って連続、最初の軸が最外周
            // ColMajorは最初の軸に沿って連続、最後の軸が最外周
            constexpr Eigen::Index outerDimIndex = (isRowMajorType ? 0 : traits::NumDimensions-1);
            constexpr Eigen::Index dimIndexOffset = (isRowMajorType ? 1 : 0);

            cereal::size_type size;
            ar( ::cereal::make_size_tag( size ) );
            assert(size==m.sizes()[outerDimIndex]);

            Eigen::DSizes<Eigen::Index, traits::NumDimensions-1> newDimensions;
            Eigen::DSizes<Eigen::Index, traits::NumDimensions> startIndices;
            Eigen::DSizes<Eigen::Index, traits::NumDimensions> sliceSizes=m.sizes();
            for(Eigen::Index i=0; i<traits::NumDimensions-1; ++i){
                newDimensions[i]=m.sizes()[i+dimIndexOffset];
                startIndices[i+dimIndexOffset]=0;
            }
            sliceSizes[outerDimIndex]=1;
            for(Eigen::Index i=0; i<m.sizes()[outerDimIndex]; ++i){
                startIndices[outerDimIndex]=i;
                ar(m.slice(startIndices,sliceSizes).reshape(newDimensions));
            }
        }
    }
    //
    // Eigen::TensorReshapingOp
    //
    template<class Archive, typename NewDimensions, typename XprType>
    inline void save_without_swap( Archive & ar, Eigen::TensorReshapingOp<NewDimensions, XprType> const & m )
    {
        using Type = Eigen::TensorReshapingOp<NewDimensions, XprType>;
        using traits=Eigen::internal::traits<Type>;
        constexpr bool isRowMajorType = static_cast<int>(traits::Layout) == static_cast<int>(Eigen::RowMajor);
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;

        if constexpr (Type::NumDimensions==1) {
            ar( ::cereal::make_size_tag( static_cast<::cereal::size_type>(m.dimensions()[0]) ) );
            auto evaluated=Eigen::TensorEvaluator(m,Eigen::DefaultDevice());
            for(Eigen::Index i=0;i<m.dimensions()[0];++i){
                ar(evaluated.coeff(i));
            }
        }else{
            // RowMajorは最後の軸に沿って連続、最初の軸が最外周
            // ColMajorは最初の軸に沿って連続、最後の軸が最外周
            constexpr Eigen::Index outerDimIndex = (isRowMajorType ? 0 : traits::NumDimensions-1);
            constexpr Eigen::Index dimIndexOffset = (isRowMajorType ? 1 : 0);

            ar( ::cereal::make_size_tag( static_cast<::cereal::size_type>(m.dimensions()[outerDimIndex]) ) );
            Eigen::DSizes<Eigen::Index, traits::NumDimensions-1> newDimensions;
            Eigen::DSizes<Eigen::Index, traits::NumDimensions> startIndices;
            Eigen::DSizes<Eigen::Index, traits::NumDimensions> sliceSizes=m.dimensions();
            for(Eigen::Index i=0; i<traits::NumDimensions-1; ++i){
                newDimensions[i]=m.dimensions()[i+dimIndexOffset];
                startIndices[i+dimIndexOffset]=0;
            }
            sliceSizes[outerDimIndex]=1;
            for(Eigen::Index i=0; i<m.dimensions()[outerDimIndex]; ++i){
                startIndices[outerDimIndex]=i;
                ar(m.slice(startIndices,sliceSizes).reshape(newDimensions));
            }
        }
    }
    template<class Archive, typename NewDimensions, typename XprType>
    inline void load_without_swap( Archive & ar, Eigen::TensorReshapingOp<NewDimensions, XprType> & m )
    {
        using Type = Eigen::TensorReshapingOp<NewDimensions, XprType>;
        using traits=Eigen::internal::traits<Type>;
        constexpr bool isRowMajorType = static_cast<int>(traits::Layout) == static_cast<int>(Eigen::RowMajor);
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;

        if constexpr (traits::NumDimensions==1) {
            cereal::size_type size;
            ar( cereal::make_size_tag( size ) );
            assert(size==m.dimensions()[0]);
            auto evaluated=Eigen::TensorEvaluator(m,Eigen::DefaultDevice());
            for(Eigen::Index i=0;i<size;++i){
                ar(evaluated.coeffRef(i));
            }
        }else{
            // RowMajorは最後の軸に沿って連続、最初の軸が最外周
            // ColMajorは最初の軸に沿って連続、最後の軸が最外周
            constexpr Eigen::Index outerDimIndex = (isRowMajorType ? 0 : traits::NumDimensions-1);
            constexpr Eigen::Index dimIndexOffset = (isRowMajorType ? 1 : 0);

            cereal::size_type size;
            ar( ::cereal::make_size_tag( size ) );
            assert(size==m.dimensions()[outerDimIndex]);

            Eigen::DSizes<Eigen::Index, traits::NumDimensions-1> newDimensions;
            Eigen::DSizes<Eigen::Index, traits::NumDimensions> startIndices;
            Eigen::DSizes<Eigen::Index, traits::NumDimensions> sliceSizes=m.dimensions();
            for(Eigen::Index i=0; i<traits::NumDimensions-1; ++i){
                newDimensions[i]=m.dimensions()[i+dimIndexOffset];
                startIndices[i+dimIndexOffset]=0;
            }
            sliceSizes[outerDimIndex]=1;
            for(Eigen::Index i=0; i<m.dimensions()[outerDimIndex]; ++i){
                startIndices[outerDimIndex]=i;
                ar(m.slice(startIndices,sliceSizes).reshape(newDimensions));
            }
        }
    }

    template< class EigenType>
    struct inner_eigen_type_wrapper{
        using StoredType = std::conditional_t<std::is_lvalue_reference_v<EigenType>,EigenType,std::decay_t<EigenType>>;
        inner_eigen_type_wrapper(EigenType&& m_):m(std::forward<EigenType>(m_)){}
        template <class Archive>
        inline void save(Archive & ar) const{
            save_without_swap(ar,m);
        }
        template <class Archive>
        inline void load(Archive & ar){
            load_without_swap(ar,m);
        }
        StoredType m;
    };
    template< class EigenType>
    inline inner_eigen_type_wrapper<EigenType> make_inner_eigen_type_wrapper(EigenType&& m_){
        return inner_eigen_type_wrapper<EigenType>(std::forward<EigenType>(m_));
    }

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

namespace cereal{
    //
    // Eigen::DenseBase
    //
    template <class Archive, ::asrc::core::traits::matrix MatrixType>
    inline void CEREAL_SAVE_FUNCTION_NAME( Archive & ar, MatrixType const & m )
    {
        using MatrixTraits=Eigen::internal::traits<MatrixType>;
        constexpr bool isRowMajorType = MatrixType::IsRowMajor;
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;

        constexpr bool always_rowmajor = is_text_archive && ::asrc::core::util::serialize_eigen_as_text_always_rowmajor;
        constexpr bool save_ndim = !::asrc::core::traits::same_archive_as<Archive,::asrc::core::util::NLJSONOutputArchive>;
        constexpr bool use_nvp = (!always_rowmajor || save_ndim);

        // numpy.arrayとの互換性のため、NLJSON以外はdtypeとndimを出力する
        if constexpr (save_ndim){
            ::asrc::core::util::save_dtype_helper<typename MatrixTraits::Scalar>(ar);

            if constexpr (MatrixTraits::RowsAtCompileTime==1 || MatrixTraits::ColsAtCompileTime==1) {
                pybind11::ssize_t ndim=1;
                ar(::cereal::make_nvp("ndim",ndim));
            }else{
                pybind11::ssize_t ndim=2;
                ar(::cereal::make_nvp("ndim",ndim));
            }
        }

        if constexpr (!always_rowmajor){
            // つねにRowMajorを強制するのでなければLayOut情報の出力が必要
            ar(::cereal::make_nvp("isRowMajorSrc",isRowMajorType));            
        }

        if constexpr (MatrixTraits::RowsAtCompileTime==1 || MatrixTraits::ColsAtCompileTime==1) {
            if constexpr (use_nvp){
                ar(::cereal::make_nvp("value",::asrc::core::util::make_inner_eigen_type_wrapper(m)));
            }else{
                ::asrc::core::util::save_without_swap(ar,m);
            }
        }else{
            if constexpr (always_rowmajor && !isRowMajorType){
                // ColMajorのMatrixを強制的にRowMajorで出力
                constexpr int newLayOut = (isRowMajorType ? Eigen::ColMajor : Eigen::RowMajor);
                Eigen::Matrix<
                    typename MatrixTraits::Scalar,
                    MatrixTraits::RowsAtCompileTime,
                    MatrixTraits::ColsAtCompileTime,
                    newLayOut,
                    MatrixTraits::MaxRowsAtCompileTime,
                    MatrixTraits::MaxColsAtCompileTime
                > swapped=m;
                if constexpr (use_nvp){
                    ar(::cereal::make_nvp("value",::asrc::core::util::make_inner_eigen_type_wrapper(swapped)));
                }else{
                    ::asrc::core::util::save_without_swap(ar,swapped);
                }
            }else{
                // as-is
                if constexpr (use_nvp){
                    ar(::cereal::make_nvp("value",::asrc::core::util::make_inner_eigen_type_wrapper(m)));
                }else{
                    ::asrc::core::util::save_without_swap(ar,m);
                }
            }
        }
    }

    template <class Archive, ::asrc::core::traits::matrix MatrixType>
    requires (
        !std::is_const_v<MatrixType>
    )
    inline void CEREAL_LOAD_FUNCTION_NAME( Archive & ar, MatrixType & m )
    {
        using MatrixTraits=Eigen::internal::traits<MatrixType>;
        constexpr bool isRowMajorType = MatrixType::IsRowMajor;
        constexpr bool is_text_archive = cereal::traits::is_text_archive<Archive>::value;

        constexpr bool always_rowmajor = is_text_archive && ::asrc::core::util::serialize_eigen_as_text_always_rowmajor;
        constexpr bool save_ndim = !::asrc::core::traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>;
        constexpr bool use_nvp = (!always_rowmajor || save_ndim);

        // numpy.arrayとの互換性のため、NLJSON以外はdtypeとndimを読み込む
        if constexpr (save_ndim){
            ::asrc::core::util::load_dtype_helper(ar);

            pybind11::ssize_t srcNdim;
            ar(::cereal::make_nvp("ndim",srcNdim));
            if constexpr (MatrixTraits::RowsAtCompileTime==1 || MatrixTraits::ColsAtCompileTime==1) {
                assert(srcNdim==1);
            }else{
                assert(srcNdim==2);
            }
        }

        bool isRowMajorSrc;
        if constexpr (!always_rowmajor){
            // つねにRowMajorを強制するのでなければLayOut情報の読み込みが必要
            ar(::cereal::make_nvp("isRowMajorSrc",isRowMajorSrc));       
        }else{
            isRowMajorSrc=true;
        }

        if constexpr (MatrixTraits::RowsAtCompileTime==1 || MatrixTraits::ColsAtCompileTime==1) {
            ::asrc::core::util::load_without_swap(ar,m);
        }else{
            if (isRowMajorType == isRowMajorSrc) {
                // as-is
                if constexpr (use_nvp){
                    ar(::cereal::make_nvp("value",::asrc::core::util::make_inner_eigen_type_wrapper(m)));
                }else{
                    ::asrc::core::util::load_without_swap(ar,m);
                }
            }else{
                // swap
                constexpr int newLayOut = (isRowMajorType ? Eigen::ColMajor : Eigen::RowMajor);
                Eigen::Matrix<
                    typename MatrixTraits::Scalar,
                    MatrixTraits::RowsAtCompileTime,
                    MatrixTraits::ColsAtCompileTime,
                    newLayOut,
                    MatrixTraits::MaxRowsAtCompileTime,
                    MatrixTraits::MaxColsAtCompileTime
                > swapped;
                if constexpr (use_nvp){
                    ar(::cereal::make_nvp("value",::asrc::core::util::make_inner_eigen_type_wrapper(swapped)));
                }else{
                    ::asrc::core::util::load_without_swap(ar,swapped);
                }

                m=std::move(swapped);
            }
        }
    }

    //
    // Eigen::TensorBase
    //
    template<class Archive, ::asrc::core::traits::tensor TensorType>
    inline void CEREAL_SAVE_FUNCTION_NAME( Archive & ar, TensorType const & m )
    {
        using TensorTraits=Eigen::internal::traits<TensorType>;
        constexpr bool isRowMajorType = static_cast<int>(TensorTraits::Layout) == static_cast<int>(Eigen::RowMajor);
        constexpr bool is_text_archive = ::cereal::traits::is_text_archive<Archive>::value;

        constexpr bool always_rowmajor = is_text_archive && ::asrc::core::util::serialize_eigen_as_text_always_rowmajor;
        constexpr bool save_ndim = !::asrc::core::traits::same_archive_as<Archive,::asrc::core::util::NLJSONOutputArchive>;
        constexpr bool use_nvp = (!always_rowmajor || save_ndim);

        // numpy.arrayとの互換性のため、NLJSON以外はdtypeとndimを出力する
        if constexpr (save_ndim){
            ::asrc::core::util::save_dtype_helper<typename TensorTraits::Scalar>(ar);

            pybind11::ssize_t ndim=TensorType::NumDimensions;
            ar(::cereal::make_nvp("ndim",ndim));
        }

        if constexpr (!always_rowmajor){
            // つねにRowMajorを強制するのでなければLayOut情報の出力が必要
            ar(::cereal::make_nvp("isRowMajorSrc",isRowMajorType));            
        }

        if constexpr (TensorType::NumDimensions==1) {
            if constexpr (use_nvp){
                ar(::cereal::make_nvp("value",::asrc::core::util::make_inner_eigen_type_wrapper(m)));
            }else{
                ::asrc::core::util::save_without_swap(ar,m);
            }
        }else{
            if constexpr (
                (is_text_archive && ::asrc::core::util::serialize_eigen_as_text_always_rowmajor)
                && !isRowMajorType
            ){
                // ColMajorをRowMajorとして出力
                // おそらくswapしたtensorを複製した方が速い
                Eigen::DSizes<Eigen::Index, TensorTraits::NumDimensions> shuffle;
                for(Eigen::Index i=0; i<TensorTraits::NumDimensions; ++i){
                    shuffle[i]=TensorTraits::NumDimensions-1-i;
                }
                Eigen::Tensor<typename TensorTraits::Scalar, TensorTraits::NumDimensions, Eigen::RowMajor, typename TensorTraits::Index> swapped=m.swap_layout().shuffle(shuffle);
                if constexpr (use_nvp){
                    ar(::cereal::make_nvp("value",::asrc::core::util::make_inner_eigen_type_wrapper(swapped)));
                }else{
                    ::asrc::core::util::save_without_swap(ar,swapped);
                }
            }else{
                // as-is
                if constexpr (use_nvp){
                    ar(::cereal::make_nvp("value",::asrc::core::util::make_inner_eigen_type_wrapper(m)));
                }else{
                    ::asrc::core::util::save_without_swap(ar,m);
                }
            }
        }
    }

    template<class Archive, ::asrc::core::traits::tensor TensorType>
    requires (
        !std::is_const_v<TensorType>
    )
    inline void CEREAL_LOAD_FUNCTION_NAME( Archive & ar, TensorType & m )
    {
        using namespace ::asrc::core::util;
        using TensorTraits=Eigen::internal::traits<TensorType>;
        constexpr bool isRowMajorType = static_cast<int>(TensorTraits::Layout) == static_cast<int>(Eigen::RowMajor);
        constexpr bool is_text_archive = ::cereal::traits::is_text_archive<Archive>::value;

        constexpr bool always_rowmajor = is_text_archive && ::asrc::core::util::serialize_eigen_as_text_always_rowmajor;
        constexpr bool save_ndim = !::asrc::core::traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>;
        constexpr bool use_nvp = (!always_rowmajor || save_ndim);

        // numpy.arrayとの互換性のため、NLJSON以外はdtypeとndimを読み込む
        if constexpr (save_ndim){
            ::asrc::core::util::load_dtype_helper(ar);

            pybind11::ssize_t srcNdim;
            ar(::cereal::make_nvp("ndim",srcNdim));
            assert(srcNdim==TensorTraits::NumDimensions);
        }

        bool isRowMajorSrc;
        if constexpr (!always_rowmajor){
            // つねにRowMajorを強制するのでなければLayOut情報の読み込みが必要
            ar(::cereal::make_nvp("isRowMajorSrc",isRowMajorSrc));       
        }else{
            isRowMajorSrc=true;
        }

        if constexpr (TensorTraits::NumDimensions==1) {
            ::asrc::core::util::load_without_swap(ar,m);
        }else{
            if (isRowMajorType == isRowMajorSrc) {
                // as-is
                if constexpr (use_nvp){
                    ar(::cereal::make_nvp("value",::asrc::core::util::make_inner_eigen_type_wrapper(m)));
                }else{
                    ::asrc::core::util::load_without_swap(ar,m);
                }
            }else{
                // swap
                constexpr int newLayOut = (isRowMajorType ? Eigen::ColMajor : Eigen::RowMajor);
                Eigen::Tensor<typename TensorTraits::Scalar, TensorTraits::NumDimensions, newLayOut, typename TensorTraits::Index> swapped;
                if constexpr (use_nvp){
                    ar(::cereal::make_nvp("value",::asrc::core::util::make_inner_eigen_type_wrapper(swapped)));
                }else{
                    ::asrc::core::util::load_without_swap(ar,swapped);
                }

                Eigen::DSizes<Eigen::Index, TensorTraits::NumDimensions> shuffle;
                for(Eigen::Index i=0; i<TensorTraits::NumDimensions; ++i){
                    shuffle[i]=TensorTraits::NumDimensions-1-i;
                }
                m=swapped.swap_layout().shuffle(shuffle);
            }
        }
    }

}
