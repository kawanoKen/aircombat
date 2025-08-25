// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// boost::uuids::uuidをシリアライズしたりpy::objectと相互変換するためのユーティリティ。
//
#pragma once
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/string_generator.hpp>
#include <cereal/cereal.hpp>

namespace py=pybind11;
namespace nl=nlohmann;

namespace cereal{
    // binaryの場合はunsigned charのarrayとして、textの場合は文字列表現を用いる。

    template<::asrc::core::traits::text_archive Archive>
    std::string save_minimal(const Archive & archive, const boost::uuids::uuid& m){
        return boost::uuids::to_string(m);
    }
    template<::asrc::core::traits::text_archive Archive>
    void load_minimal(const Archive & archive, boost::uuids::uuid& m, const std::string& value){
        m=boost::uuids::string_generator()(value);
    }
    template<::asrc::core::traits::non_text_archive Archive>
    void serialize(Archive & archive, boost::uuids::uuid& m){
        if constexpr (BOOST_VERSION < 108600) {
            archive(m.data);
        }else{
            archive( make_size_tag( static_cast<::cereal::size_type>(m.size()) ) );
            archive( binary_data( m.data(), m.size() * sizeof(std::uint8_t) ) );
        }
    }
}

NLOHMANN_JSON_NAMESPACE_BEGIN
    // jsonへは文字列表現を用いて変換

    template<>
    struct adl_serializer<boost::uuids::uuid> {
        static void to_json(json& j, const boost::uuids::uuid& m) {
            j=boost::uuids::to_string(m);
        }
        static void from_json(const json& j, boost::uuids::uuid& m) {
            assert(j.is_string());
            m=boost::uuids::string_generator()(j.get<std::string>());
        }
    };

NLOHMANN_JSON_NAMESPACE_END

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
    namespace detail{
        template<>
        struct type_caster<boost::uuids::uuid>
        {
        public:
            PYBIND11_TYPE_CASTER(boost::uuids::uuid, _("uuid"));
            bool load(handle src, bool)
            {
                try {
                    if constexpr (BOOST_VERSION < 106600) {
                        //古いBoost(1.66以前)ではuuids::string_generatorにおける文字列の正当性チェックに漏れがあるため、正しくない文字列でも変換が通ってしまう。
                        //Ubuntu 18.04においてaptで入れられるBoostのバージョンは1.65.1でありこのバグが残っている状態であるため、
                        //当分の間はUbuntu 18.04での使用も想定してPythonのuuid.UUIDクラスに文字列の正当性チェックを委ねることとする。
                        module_ uuid=module_::import("uuid");
                        std::string s=uuid.attr("UUID")(src.attr("__str__")()).attr("__str__")().cast<std::string>();
                        value=boost::uuids::string_generator()(s);
                    }else{
                        value=boost::uuids::string_generator()(src.attr("__str__")().cast<std::string>());
                    }
                    return true;
                }
                catch (...)
                {
                    return false;
                }
            }

            static handle cast(boost::uuids::uuid src, return_value_policy /* policy */, handle /* parent */)
            {
                module_ uuid=module_::import("uuid");
                return uuid.attr("UUID")(boost::uuids::to_string(src)).release();
            }
        };
    }
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
