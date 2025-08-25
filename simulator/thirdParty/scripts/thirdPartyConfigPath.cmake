#Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

get_filename_component(THIRDPARTY_INSTALL_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
list(APPEND CMAKE_PREFIX_PATH
     "${THIRDPARTY_INSTALL_DIR}/share/cmake/pybind11"
     "${THIRDPARTY_INSTALL_DIR}/share/eigen3/cmake"
     "${THIRDPARTY_INSTALL_DIR}/lib/cmake/nlopt"
     "${THIRDPARTY_INSTALL_DIR}/share/cmake/nlohmann_json"
     "${THIRDPARTY_INSTALL_DIR}/lib/cmake/magic_enum"
     "${THIRDPARTY_INSTALL_DIR}/lib/cmake/cereal"
)
set(BOOST_ROOT "${THIRDPARTY_INSTALL_DIR}" CACHE PATH "Preferred installation prefix of Boost" FORCE)
