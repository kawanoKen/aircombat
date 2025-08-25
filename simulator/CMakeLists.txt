# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
cmake_minimum_required(VERSION 3.10)
cmake_policy(PUSH)
cmake_policy(VERSION 3.10...3.29)
project(${TARGET_NAME})
if(${WITH_THIRDPARTIES})
    if(EXISTS "${PROJECT_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}/thirdParty/install/thirdPartyConfigPath.cmake")
        include("${PROJECT_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}/thirdParty/install/thirdPartyConfigPath.cmake")
    endif()
endif()
set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set(PythonLibsNew_FIND_VERSION $ENV{PythonLibsNew_FIND_VERSION})
set(Python_ROOT_DIR "${Python3_INCLUDE_DIRS}/..")
set(Python3_ROOT_DIR "${Python3_INCLUDE_DIRS}/..")
find_package(Python $ENV{PythonLibsNew_FIND_VERSION} COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(NLopt REQUIRED)
find_package(Boost REQUIRED)
message(STATUS "Include:${Python3_INCLUDE_DIRS}")
message(STATUS "Libs:${Python3_LIBRARY_DIRS}")
message(STATUS "Numpy:${Python3_Numpy_INCLUDE_DIRS}")
message(STATUS "Eigen:${EIGEN3_INCLUDE_DIR}")
message(STATUS "Boost:${Boost_INCLUDE_DIRS}")
message(STATUS "Pybind11:${pybind11_INCLUDE_DIR}")
set(CMAKE_POSITION_INDEPENDENT_CODE True)
include_directories(
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/thirdParty/include"
    "${EIGEN3_INCLUDE_DIR}"
    "${EIGEN3_INCLUDE_DIR}/unsupported"
    "${Boost_INCLUDE_DIRS}"
    "${Python3_INCLUDE_DIRS}"
    "${Python3_Numpy_INCLUDE_DIRS}"
    "${pybind11_INCLUDE_DIR}"
)
link_directories(
    "${Python3_LIBRARY_DIRS}"
)
if(NOT "${MSYS_ROOT_DIR}" STREQUAL "")
    #MSYS2 seems it does not add /usr/local by default.
    include_directories(
        "${MSYS_ROOT_DIR}usr/local/include"
    )
    link_directories(
        "${MSYS_ROOT_DIR}usr/local/lib"
    )
endif()

add_definitions(
    -DEIGEN_MPL2_ONLY
    -D_USE_MATH_DEFINES
    -DHAVE_SNPRINTF
    -DMODULE_NAME=Core
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -ftemplate-backtrace-limit=0")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftemplate-backtrace-limit=0")
endif()
if(MSVC)
    add_compile_options("/utf-8")
    add_compile_options("/bigobj")
    add_compile_options("/D_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING")
    add_definitions(
        -DNOMINMAX
    )
endif()
file(GLOB_RECURSE source "${PROJECT_SOURCE_DIR}/src/**.cpp" "${PROJECT_SOURCE_DIR}/src/**.c")
pybind11_add_module(Core SHARED
    ${source}
)
if(WIN32)
    if(MSVC)
        set_target_properties(Core PROPERTIES PREFIX "" SUFFIX ".dll")
    else()
        set_target_properties(Core PROPERTIES PREFIX "lib" SUFFIX ".dll")
    endif()
else()
    set_target_properties(Core PROPERTIES PREFIX "lib" SUFFIX ".so")
endif()
if(${WITH_THIRDPARTIES})
    if(NOT WIN32)
        if(APPLE)
            set_target_properties(Core PROPERTIES
                BUILD_WITH_INSTALL_RPATH FALSE
                LINK_FLAGS "-Wl,-rpath,@loader_path/../thirdParty/lib"
            )
        else()
            set_target_properties(Core PROPERTIES
                BUILD_WITH_INSTALL_RPATH FALSE
                LINK_FLAGS "-Wl,-rpath,$ORIGIN/../thirdParty/lib"
            )
        endif()
    endif()
endif()
target_link_libraries(Core PRIVATE
    NLopt::nlopt
)
if(WIN32)
    target_link_libraries(Core PRIVATE
        bcrypt
    )
else()
    target_link_libraries(Core PRIVATE
        m
    )
endif()

#----------------------------------------------------------------
# installation after build
#----------------------------------------------------------------
install(
    TARGETS Core
    DESTINATION ${TARGET_NAME}/core
)
if(EXISTS "${PROJECT_SOURCE_DIR}/include")
    install(
        DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
        DESTINATION "${PROJECT_SOURCE_DIR}/${TARGET_NAME}/include/${TARGET_NAME}"
    )
endif()
if(${WITH_THIRDPARTIES})
    install(
        DIRECTORY "${PROJECT_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}/thirdParty/install/"
        DESTINATION "${PROJECT_SOURCE_DIR}/${TARGET_NAME}/thirdParty"
    )
endif()

cmake_policy(POP)
