/**
 * Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief Entity を一意に識別するためのidentifier
 * 
 * @details シミュレーション登場物を表すEntity、Entityを管理するEntityManager、Entityへのアクセス制限を担うEntityAccessorの3つからなる。
 */
#pragma once
#include "Common.h"
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>
#include <limits>
#include <memory>
#include <sstream>
#include <boost/functional/hash.hpp>
#include "Pythonable.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

/**
 * @brief Factoryで生成されるEntityインスタンスを一意に識別するためのクラス。
 * 
 * @details 以下の4つの32bit符号付き整数値から構成される。
 *      - worker_index　… シミュレーション管理プロセスの識別( Factory , EntityManager クラスオブジェクトの識別)
 *      - vector_index　… シミュレーションインスタンスの識別( Factory , EntityManager インスタンスの識別)(-1(0xFFFFFFFF)はインスタンス非依存あるいはダミーであることを表す)
 *      - episode_index … シミュレーションエピソードの識別 負数はエピソード非依存あるいはダミーであることを表す)
 *      - entity_index … コンテキスト( worker_index , vector_index , episode_index )内における Entity の識別 (負数は無効な EntityIdentifier であることを表す)
 * 
 *      worker_index は EntityManager のstaticメンバとして管理し、プロセスごとに異なる値を設定する。 SimulationManager の引数である worker_index と連動する。変数の名称はray.rllibのEnvContextに準拠したもの。
 * 
 *      vector_index は EntityManager インスタンスのメンバとして管理し、コンストラクタで設定する。 EntityManager や SimulationManager の引数である vector_index と連動する。変数の名称ray.rllibのEnvContextに準拠したもの。
 * 
 *      episode_index は EntityManager や SimulationManager インスタンスのメンバとして、エピソードごとにインクリメントする。初期値は0。
 */
ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS(EntityIdentifier)
    static constexpr std::int32_t invalid_vector_index=-1; //!< 無効な vector_index を表す定数
    static constexpr std::int32_t invalid_episode_index=-1; //!< 無効な episode_index を表す定数
    static constexpr std::int32_t invalid_entity_index=-1; //!< 無効な entity_index を表す定数
    std::int32_t worker_index; //!< シミュレーション管理プロセスの識別( Factory , EntityManager クラスオブジェクトの識別)
    std::int32_t vector_index; //!< シミュレーションインスタンスの識別( Factory , EntityManager インスタンスの識別)(-1(0xFFFFFFFF)はインスタンス非依存あるいはダミーであることを表す)
    std::int32_t episode_index; //!< シミュレーションエピソードの識別 負数はエピソード非依存あるいはダミーであることを表す)
    std::int32_t entity_index; //!< コンテキスト( worker_index , vector_index , episode_index )内における Entity の識別 (負数は無効な EntityIdentifier であることを表す)
    EntityIdentifier();
    /**
     * @brief 4種の整数を個別に与えるコンストラクタ
     */
    EntityIdentifier(const std::int32_t& w, const std::int32_t& v, const std::int32_t& ep, const std::int32_t& en);
    /**
     * @brief jsonから生成するコンストラクタ
     * 
     * @code
     *  {
     *      "w": 0,
     *      "v": 0,
     *      "ep": 0,
     *      "en": 0
     *  }
     * @endcode
     */
    EntityIdentifier(const nl::json& j);
    std::string toString() const;
    template<class Archive>
    void save(Archive & archive) const{
        archive(
            cereal::make_nvp("w",worker_index),
            cereal::make_nvp("v",vector_index),
            cereal::make_nvp("ep",episode_index),
            cereal::make_nvp("en",entity_index)
        );
    }
    template<class Archive>
    void load(Archive & archive) {
        if constexpr (::asrc::core::traits::same_archive_as<Archive,::asrc::core::NLJSONInputArchive>){
            const nl::json& j=archive.getCurrentNode();
            if(j.is_array()){
                if(j.size()==4){
                    worker_index=j.at(0);
                    vector_index=j.at(1);
                    episode_index=j.at(2);
                    entity_index=j.at(3);
                    return;
                }else{
                    throw std::runtime_error("EntityIdentifier only accepts 4-element json array or object.");
                }
            }else if(!j.is_object()){
                throw std::runtime_error("EntityIdentifier only accepts 4-element json array or object.");
            }
        }
        if(!asrc::core::util::try_load(archive,"w",worker_index)){
            worker_index=getWorkerIndex();
        }
        if(!asrc::core::util::try_load(archive,"v",vector_index)){
            vector_index=-1;
        }
        if(!asrc::core::util::try_load(archive,"ep",episode_index)){
            episode_index=-1;
        }
        if(!asrc::core::util::try_load(archive,"en",entity_index)){
            entity_index=-1;
        }
    }
    private:
    static std::int32_t getWorkerIndex();
};

inline bool operator==(const EntityIdentifier& lhs,const EntityIdentifier& rhs) noexcept{
    return (
        lhs.worker_index == rhs.worker_index &&
        lhs.vector_index == rhs.vector_index &&
        lhs.episode_index == rhs.episode_index &&
        lhs.entity_index == rhs.entity_index
    );
}
inline bool operator!=(const EntityIdentifier& lhs,const EntityIdentifier& rhs) noexcept{
    return !(lhs==rhs);
}
inline bool operator<(const EntityIdentifier& lhs,const EntityIdentifier& rhs) noexcept{
    return (
        lhs.worker_index < rhs.worker_index ||
        (lhs.worker_index == rhs.worker_index && (
            lhs.vector_index < rhs.vector_index ||
            (lhs.vector_index == rhs.vector_index && (
                lhs.episode_index < rhs.episode_index ||
                (lhs.episode_index == rhs.episode_index &&
                    lhs.entity_index < rhs.entity_index
                )
            ))
        ))
    );
}
inline bool operator>(const EntityIdentifier& lhs,const EntityIdentifier& rhs) noexcept{
    return (
        lhs.worker_index > rhs.worker_index ||
        (lhs.worker_index == rhs.worker_index && (
            lhs.vector_index > rhs.vector_index ||
            (lhs.vector_index == rhs.vector_index && (
                lhs.episode_index > rhs.episode_index ||
                (lhs.episode_index == rhs.episode_index &&
                    lhs.entity_index > rhs.entity_index
                )
            ))
        ))
    );
}
inline bool operator<=(const EntityIdentifier& lhs,const EntityIdentifier& rhs) noexcept{
    return !(lhs>rhs);
}
inline bool operator>=(const EntityIdentifier& lhs,const EntityIdentifier& rhs) noexcept{
    return !(lhs<rhs);
}
inline std::ostream& operator<<(std::ostream& os, const EntityIdentifier& ei){
    os << "(w:" << ei.worker_index << ",";
    os << "v:" << ei.vector_index << ",";
    os << "ep:" << ei.episode_index << ",";
    os << "en:" << ei.entity_index << ")";
    return os;
}

inline void to_json(nl::json& j,const EntityIdentifier& ei){
    j = ei.to_json();
}

inline void from_json(const nl::json& j,EntityIdentifier& ei){
    ei.load_from_json(j);
}
void exportEntityIdentifier(py::module &m);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

template<>
struct std::hash<::asrc::core::EntityIdentifier>{
    std::size_t operator()(const ::asrc::core::EntityIdentifier& ei) const noexcept{
        std::size_t ret=0;
        boost::hash_combine(ret,ei.worker_index);
        boost::hash_combine(ret,ei.vector_index);
        boost::hash_combine(ret,ei.episode_index);
        boost::hash_combine(ret,ei.entity_index);
        return ret;
    }
};

ASRC_PYBIND11_MAKE_OPAQUE(std::set<::asrc::core::EntityIdentifier>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<::asrc::core::EntityIdentifier>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,::asrc::core::EntityIdentifier>);
ASRC_PYBIND11_MAKE_OPAQUE(std::pair<::asrc::core::EntityIdentifier,bool>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::pair<::asrc::core::EntityIdentifier,bool>>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<::asrc::core::EntityIdentifier,boost::uuids::uuid>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<boost::uuids::uuid,::asrc::core::EntityIdentifier>);
