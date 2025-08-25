# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#----------------------------------------------------------------
# CMakeLists.txt template for ASRCAISim1's user_plugin
#----------------------------------------------------------------
cmake_minimum_required(VERSION 3.10)
cmake_policy(PUSH)
cmake_policy(VERSION 3.10...3.29)
project(${TARGET_NAME}${BUILD_IDENTIFIER})
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
#----------------------------------------------------------------
# find packages
#----------------------------------------------------------------
set(PythonLibsNew_FIND_VERSION $ENV{PythonLibsNew_FIND_VERSION})
set(Python_ROOT_DIR "${Python3_INCLUDE_DIRS}/..")
set(Python3_ROOT_DIR "${Python3_INCLUDE_DIRS}/..")
find_package(ASRCAISim1 REQUIRED) # put 1st
find_package(Python $ENV{PythonLibsNew_FIND_VERSION} COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
list(GET ASRC_PATH_INFO 0 ASRC_Python3_INCLUDE_DIRS)
list(GET ASRC_PATH_INFO 1 ASRC_Python3_LIBRARY_DIRS)
list(GET ASRC_PATH_INFO 2 ASRC_Python3_Numpy_INCLUDE_DIRS)
list(GET ASRC_PATH_INFO 3 ASRC_CORE_SIMULATOR_DIR)
message(STATUS "Python Include:${ASRC_Python3_INCLUDE_DIRS}")
message(STATUS "Python Libs:${ASRC_Python3_LIBRARY_DIRS}")
message(STATUS "Python Executable:${Python3_EXECUTABLE}")
message(STATUS "Numpy:${ASRC_Python3_Numpy_INCLUDE_DIRS}")
message(STATUS "Eigen:${EIGEN3_INCLUDE_DIR}")
message(STATUS "Boost:${Boost_INCLUDE_DIRS}")
message(STATUS "Pybind11:${pybind11_INCLUDE_DIR}")
message(STATUS "Core Simulator Dir:${ASRC_CORE_SIMULATOR_DIR}")

#----------------------------------------------------------------
# check the version of required plugins
#----------------------------------------------------------------
set(_USER_PLUGIN_NAMES)
foreach(_PLUGIN_DIR ${ASRC_USER_PLUGIN_DIRS})
    get_filename_component(_PLUGIN_NAME ${_PLUGIN_DIR} NAME_WE)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${_PLUGIN_DIR}/share/cmake)
    find_package(${_PLUGIN_NAME} REQUIRED)
    list(APPEND _USER_PLUGIN_NAMES ${_PLUGIN_NAME})
    message("${${_PLUGIN_NAME}_INCLUDE_DIRS}")
endforeach()

#----------------------------------------------------------------
# target configuration
#----------------------------------------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE True)
include_directories(
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/thirdParty/include"
    "${EIGEN3_INCLUDE_DIR}"
    "${EIGEN3_INCLUDE_DIR}/unsupported"
    "${Boost_INCLUDE_DIRS}"
    "${ASRC_Python3_INCLUDE_DIRS}"
    "${ASRC_Python3_Numpy_INCLUDE_DIRS}"
    "${pybind11_INCLUDE_DIR}"
)
link_directories(
    "${ASRC_Python3_LIBRARY_DIRS}"
)
if(NOT ${MSYS_ROOT_DIR} STREQUAL "")
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
    -DMODULE_NAME=${TARGET_NAME}
)
if(DEFINED BUILD_IDENTIFIER)
    link_directories(
        "${PROJECT_SOURCE_DIR}/${TARGET_NAME}"
    )
    add_definitions(
        -DMODULE_BUILD_IDENTIFIER=${BUILD_IDENTIFIER}
    )
else()
    # the parts independent of build identifier should be built iff built without build identifier
    add_definitions(
        -DASRC_BUILD_PLUGIN_BII_PART
    )
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
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
pybind11_add_module(${CMAKE_PROJECT_NAME} SHARED
    ${source}
)
if(WIN32)
    if(MSVC)
        set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".dll")
    else()
        set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES PREFIX "lib" SUFFIX ".dll")
    endif()
else()
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES PREFIX "lib" SUFFIX ".so")
endif()
if(${WITH_THIRDPARTIES})
    if(NOT WIN32)
        if(APPLE)
            set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
                BUILD_WITH_INSTALL_RPATH FALSE
                LINK_FLAGS "-Wl,-rpath,@loader_path/thirdParty/lib"
            )
        else()
            set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
                BUILD_WITH_INSTALL_RPATH FALSE
                LINK_FLAGS "-Wl,-rpath,$ORIGIN/thirdParty/lib"
            )
        endif()
    endif()
endif()
#----------------------------------------------------------------
# link libraries
#----------------------------------------------------------------
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
	ASRCAISim1::Core
)
foreach(_PLUGIN_NAME ${ASRC_CORE_PLUGIN_NAMES})
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
        ASRCAISim1::${_PLUGIN_NAME}
    )
endforeach()
if(NOT WIN32)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
        m
    )
endif()

message(STATUS "Plugins: ${_USER_PLUGIN_NAMES}")
if(DEFINED BUILD_IDENTIFIER)
    foreach(_PLUGIN_NAME ${_USER_PLUGIN_NAMES})
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
            "${_PLUGIN_NAME}::All_B"
        )
        message(STATUS "${_PLUGIN_NAME}")
    endforeach()
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
        ${TARGET_NAME}
    )
else()
    foreach(_PLUGIN_NAME ${_USER_PLUGIN_NAMES})
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
            "${_PLUGIN_NAME}::All"
        )
        message(STATUS "${_PLUGIN_NAME}")
    endforeach()
endif()

#----------------------------------------------------------------
# installation after build
#----------------------------------------------------------------
install(
    TARGETS ${CMAKE_PROJECT_NAME}
		DESTINATION ${TARGET_NAME}
)
if(EXISTS "${PROJECT_SOURCE_DIR}/include")
    if(DEFINED BUILD_IDENTIFIER)
        install(
            DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
            DESTINATION "${PROJECT_SOURCE_DIR}/${TARGET_NAME}/include/${BUILD_IDENTIFIER}/${TARGET_NAME}"
        )
    else()
        install(
            DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
            DESTINATION "${PROJECT_SOURCE_DIR}/${TARGET_NAME}/include/${TARGET_NAME}"
        )
    endif()
endif()

if(${WITH_THIRDPARTIES})
    install(
        DIRECTORY "${PROJECT_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}/thirdParty/install/"
        DESTINATION "${PROJECT_SOURCE_DIR}/${TARGET_NAME}/thirdParty"
    )
endif()

#----------------------------------------------------------------
# add user-defined dependencies
#----------------------------------------------------------------
#find_package(a_third_party_package REQUIRED)
#target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
#    a_third_party_package::target
#)

cmake_policy(POP)
