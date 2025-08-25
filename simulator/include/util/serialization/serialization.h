// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// serialization関係の機能をまとめてインクルードするためのヘッダ
//
#pragma once
#include "common.h"
#include "enum.h"
#include "nljson.h"
#include "pyobject.h"
#include "NLJSONArchive.h"
#include "Serializer.h"
#include "eigen.h"
#include "mersenne_twister.h"
#include "uuid.h"
#include "as_reference.h"
#include "internal_state_serializer.h"
#include <cereal/types/polymorphic.hpp>

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

void exportSerialization(py::module &m);

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
