// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// serialization関係の雑多な関数群
//
#pragma once
#include <cereal/cereal.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/queue.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/external/base64.hpp>
#include "../../traits/cereal.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

// try_load
// Archive::containsで読み込み前にkeyの存否判定ができる場合は、存在する場合のみ値を読み込み、存否判定結果を返す。
// 存否判定ができないArchiveの場合は通常どおり値を読み込むもうとし、Archiveクラス依存の例外処理に委ねる。
template<traits::input_archive Archive, class KeyType, class ValueType>
bool try_load(Archive& ar, KeyType&& key, ValueType& value){
    if constexpr (traits::archive_has_contains<Archive, KeyType>) {
        if(!ar.contains(std::forward<KeyType>(key))){
            return false;
        }
    }
    ar(cereal::make_nvp(std::forward<KeyType>(key),value));
    return true;
}

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
