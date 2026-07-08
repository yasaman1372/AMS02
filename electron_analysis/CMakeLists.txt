cmake_minimum_required(VERSION 2.8)

# find acsoft first, to override compiler
if (ACsoft_DIR)
  find_package(ACsoft REQUIRED NO_DEFAULT_PATH)
else()
  find_package(ACsoft REQUIRED)
endif()

PROJECT(ElectronAnalysis)
complete_ACSoft_setup()

# set ExampleAnalysis includes
include_directories("${PROJECT_SOURCE_DIR}/Libraries/ExampleAnalysisTree")

# set LeptonAnalysisTools includes
include_directories("${PROJECT_SOURCE_DIR}/Libraries/LeptonAnalysisTools")

# doxygen target
include(Auxiliary/doxygen.cmake)

set(FOLDER_NAME "Libraries")
add_subdirectory(Libraries)

set(FOLDER_NAME "Programs")
add_subdirectory(Programs)

set(FOLDER_NAME "ROOT Macros")
add_subdirectory(Macros)
