cmake_minimum_required(VERSION 3.21)

# Allow older CMake syntax in vendored dependencies (qpOASES, Eigen)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "")

project(robot_arm_sim_cbf_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# ---------- Dependencies ----------

# pybind11
find_package(pybind11 CONFIG REQUIRED)

# Eigen3: try system first, then fetch
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found on system — fetching via FetchContent")
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
endif()

# qpOASES via FetchContent (built from source as static library)
FetchContent_Declare(
    qpOASES
    GIT_REPOSITORY https://github.com/coin-or/qpOASES.git
    GIT_TAG releases/3.2.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(qpOASES)

# ---------- Extension module ----------

pybind11_add_module(_cbf_core
    csrc/bindings.cpp
    csrc/cbf_core/barrier.cpp
    csrc/cbf_core/qp_solver.cpp
    csrc/cbf_core/cbf_solver.cpp
)

target_include_directories(_cbf_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${qpOASES_SOURCE_DIR}/include
)

target_link_libraries(_cbf_core PRIVATE
    Eigen3::Eigen
    qpOASES
)

# Release optimisations
target_compile_options(_cbf_core PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>
)

# scikit-build-core: install into the Python package directory
install(TARGETS _cbf_core DESTINATION robot_arm_sim)
