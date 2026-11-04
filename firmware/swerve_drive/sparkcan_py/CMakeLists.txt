cmake_minimum_required(VERSION 3.18)   # need >=3.11 for FetchContent; you have 3.22
project(sparkcan VERSION 1.0 LANGUAGES CXX)

# -----------------------------
# C++ settings (unchanged)
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------
# Core shared library (unchanged)
# -----------------------------
add_library(sparkcan SHARED
    src/SparkBase.cpp
    src/SparkMax.cpp
    src/SparkFlex.cpp
)

target_include_directories(sparkcan PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# -----------------------------
# Example executables (unchanged)
# -----------------------------
add_executable(example_control examples/control.cpp)
add_executable(example_pid     examples/pid.cpp)
add_executable(example_status  examples/status.cpp)

target_link_libraries(example_control PRIVATE sparkcan)
target_link_libraries(example_pid     PRIVATE sparkcan)
target_link_libraries(example_status  PRIVATE sparkcan)

# -----------------------------
# Install core C++ lib (unchanged)
# -----------------------------
install(TARGETS sparkcan
    EXPORT sparkcanTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT sparkcanTargets
    FILE sparkcanTargets.cmake
    NAMESPACE sparkcan::
    DESTINATION lib/cmake/sparkcan
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/sparkcanConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/sparkcanConfig.cmake"
    INSTALL_DESTINATION lib/cmake/sparkcan
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/sparkcanConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/sparkcanConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/sparkcanConfigVersion.cmake"
    DESTINATION lib/cmake/sparkcan
)

# =============================
#        Python bindings
# =============================
option(BUILD_PYTHON "Build Python extension module" ON)

if(BUILD_PYTHON)
  # Make pybind11 use the same Python you invoked CMake with (Conda-friendly).
  set(PYBIND11_FINDPYTHON ON)

  # Require a Python with dev headers (pip/scikit-build will pass this automatically)
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

  # Find modern pybind11; else fetch v2.12 (safe for Python 3.12+).
  find_package(pybind11 2.12 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    message(STATUS "pybind11 >= 2.12 not found; fetching v2.12.0 via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)
  endif()

  # ---- Python extension target ----
  # IMPORTANT: your binding source must define PYBIND11_MODULE(_sparkcan, m)
  pybind11_add_module(_sparkcan MODULE
      bindings/pybind11/sparkcan_bindings.cpp
  )

  target_link_libraries(_sparkcan PRIVATE sparkcan)
  target_include_directories(_sparkcan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

  # Make the extension find libsparkcan.so at runtime without env vars:
  # we install a copy of libsparkcan next to the extension & set RPATH=$ORIGIN
  set_target_properties(_sparkcan PROPERTIES
      BUILD_RPATH "\$ORIGIN"
      INSTALL_RPATH "\$ORIGIN"
  )

  # Install the extension into the Python package directory inside the wheel/site-packages
  install(TARGETS _sparkcan
      LIBRARY DESTINATION sparkcan_py     # Unix/macOS
      RUNTIME DESTINATION sparkcan_py     # Windows
  )

  # If sparkcan is shared, ship it next to the extension too (for import-time resolution)
  get_target_property(_sc_type sparkcan TYPE)
  if(_sc_type STREQUAL "SHARED_LIBRARY")
    install(TARGETS sparkcan
        LIBRARY DESTINATION sparkcan_py
        RUNTIME DESTINATION sparkcan_py)
  endif()

  # Install the pure-Python shim that re-exports the compiled module
  # Create this file at: python/sparkcan_py/__init__.py  with:  from ._sparkcan import *
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/python/sparkcan_py/__init__.py
          DESTINATION sparkcan_py)
endif()
