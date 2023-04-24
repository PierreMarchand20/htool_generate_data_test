# =============================================================================#
# ============================== Project ======================================#
# =============================================================================#
cmake_minimum_required(VERSION 3.10)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    project(
        Htool-generate-data-test
        VERSION 1.0
        LANGUAGES CXX)
else()
    project(
        Htool-generate-data-test
        VERSION 1.0
        DESCRIPTION "A header only c++ library that provides Hierarchical matrices."
        HOMEPAGE_URL "https://github.com/htool-ddm/htool"
        LANGUAGES CXX)
endif()

# To force c++11
if(${CMAKE_VERSION} VERSION_LESS 3.1)
    add_compile_options(-std=c++11)
elseif(${CMAKE_VERSION} VERSION_LESS 3.6.3 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    add_compile_options(-std=c++11)
else()
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # To set default CMAKE_BUILD_TYPE
    set(default_build_type "Release")
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
        set(CMAKE_BUILD_TYPE
            "${default_build_type}"
            CACHE STRING "Choose the type of build." FORCE)
    endif()

    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

    # # Formatting
    # include(external/htool/cmake/formatting.cmake)
    # file(
    #     GLOB_RECURSE
    #     ALL_CODE_FILES
    #     ${PROJECT_SOURCE_DIR}/include/*.[h]pp
    #     ${PROJECT_SOURCE_DIR}/include/*.[h]
    #     ${PROJECT_SOURCE_DIR}/examples/*.[ch]pp
    #     ${PROJECT_SOURCE_DIR}/examples/*.[ch]
    #     ${PROJECT_SOURCE_DIR}/tests/*.[ch]pp
    #     ${PROJECT_SOURCE_DIR}/tests/*.[ch])

    # clang_format(format ${ALL_CODE_FILES})
    # file(GLOB_RECURSE CMAKE_FILES ${PROJECT_SOURCE_DIR}/CMakeLists.txt)
    # cmake_format(cmake_format ${CMAKE_FILES})

    # Files to do find_package for some module
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_modules")

    # Information about compilation exported
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # Let's nicely support folders in IDE's
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Testing only available if this is the main app
    # Note this needs to be done in the main CMakeLists
    # since it calls enable_testing, which must be in the
    # main CMakeLists.
    include(CTest)

    # # Docs only available if this is the main app
    # find_package(Doxygen)
    # if(Doxygen_FOUND)
    # add_subdirectory(docs)
    # else()
    # message(STATUS "Doxygen not found, not building docs")
    # endif()

endif()

# =============================================================================#
# ========================== External Libraries ===============================#
# =============================================================================#
# Boost package - special functions
find_package(Boost REQUIRED)

# OPENMP
find_package(OpenMP)

# BLAS
find_package(BLAS REQUIRED)
message("-- Found Blas implementation:" "${BLAS_LIBRARIES}")

# LAPACK
find_package(LAPACK)
message("-- Found Lapack:" "${LAPACK_LIBRARIES}")

# MPI
find_package(MPI REQUIRED)
message("-- MPI libraries found in " "${MPI_LIBRARIES}")
message("-- MPI include files found in " "${MPI_INCLUDE_PATH}")

# OPENMP
find_package(OpenMP)

# =============================================================================#
# =========================== Libraries =======================================#
# =============================================================================#

add_library(Htool_generate_data_test INTERFACE)
target_include_directories(Htool_generate_data_test INTERFACE "external/BemTool" ${Boost_INCLUDE_DIRS} "external/htool/include/" ${MPI_INCLUDE_PATH} "external/hpddm/include")

target_link_libraries(Htool_generate_data_test INTERFACE ${MPI_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)

# =============================================================================#
# ========================== Repertories ======================================#
# =============================================================================#
add_subdirectory(src)
