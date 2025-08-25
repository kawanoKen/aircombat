// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// Eigenに関するtraits
//
#pragma once
#include <type_traits>
#include <concepts>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/CXX11/Tensor>
#include "../util/macros/common_macros.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_NAMESPACE_BEGIN(traits)

template<typename MatrixType>
concept matrix = std::derived_from<MatrixType,Eigen::DenseBase<MatrixType>>;

template<typename TensorType>
concept tensor = std::derived_from<TensorType,Eigen::TensorBase<TensorType,Eigen::ReadOnlyAccessors>>;

template<typename TensorType>
concept has_dimensions = requires (TensorType& t) {
    t.dimensions();
};

template<typename TensorType>
concept has_sizes = requires (TensorType& t) {
    t.sizes();
};

template<typename TensorType>
concept has_size = requires (TensorType& t) {
    { t.size() } -> std::integral;
};

ASRC_NAMESPACE_END(traits)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

