cmake_minimum_required(VERSION 3.12)

set(ACsoft_DIR "$ENV{ACSOFTDIR}/build")
find_package(ACsoft REQUIRED)

PROJECT(PhotonFramework)
complete_ACSoft_setup()

include_directories("${PROJECT_SOURCE_DIR}/Libraries/PhotonFramework")

set(FOLDER_NAME "Libraries")
add_subdirectory(Libraries)

set(FOLDER_NAME "Programs")
add_subdirectory(Programs)

#set(FOLDER_NAME "ROOT Macros")
#add_directory(Macros)
