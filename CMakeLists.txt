cmake_minimum_required(VERSION 2.8)
set(SOLUTIONTITLE LevelSets CACHE TYPE STRING)
project(${SOLUTIONTITLE})
file(GLOB FILES_SRC
    "src/*.cpp"
    "src/*.h"
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 /bigobj /EHa")
set(VVRFRAMEWORK_DIR "" CACHE PATH "Location of VVR Framework")

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${VVRFRAMEWORK_DIR}/include)
include_directories(${VVRFRAMEWORK_DIR}/include/VVRScene)
include_directories(${VVRFRAMEWORK_DIR}/include/GeoLib)
include_directories(${VVRFRAMEWORK_DIR}/include/MathGeoLib)
include_directories(${CMAKE_SOURCE_DIR}/Eigen)

add_executable(${SOLUTIONTITLE} ${FILES_SRC})

if (CMAKE_BUILD_TYPE MATCHES Debug OR
    EXISTS "${VVRFRAMEWORK_DIR}/VVRScene/Debug" AND
    EXISTS "${VVRFRAMEWORK_DIR}/GeoLib/Debug" AND
    EXISTS "${VVRFRAMEWORK_DIR}/MathGeoLib/Debug")

    target_link_libraries(
    ${SOLUTIONTITLE}
    ${VVRFRAMEWORK_DIR}/VVRScene/Debug/VVRScene_d.lib 
    ${VVRFRAMEWORK_DIR}/GeoLib/Debug/GeoLib_d.lib 
    ${VVRFRAMEWORK_DIR}/MathGeoLib/Debug/MathGeoLib_d.lib
    )

    add_custom_command(TARGET ${SOLUTIONTITLE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${VVRFRAMEWORK_DIR}/VVRScene/Debug" ${CMAKE_BINARY_DIR}/$<CONFIG>)
    add_custom_command(TARGET ${SOLUTIONTITLE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${VVRFRAMEWORK_DIR}/GeoLib/Debug" ${CMAKE_BINARY_DIR}/$<CONFIG>)
    add_custom_command(TARGET ${SOLUTIONTITLE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${VVRFRAMEWORK_DIR}/MathGeoLib/Debug" ${CMAKE_BINARY_DIR}/$<CONFIG>)
endif()
    
if (CMAKE_BUILD_TYPE MATCHES Release OR
    EXISTS "${VVRFRAMEWORK_DIR}/VVRScene/Release" AND
    EXISTS "${VVRFRAMEWORK_DIR}/GeoLib/Release" AND
    EXISTS "${VVRFRAMEWORK_DIR}/MathGeoLib/Release")

    target_link_libraries(
    ${SOLUTIONTITLE}
    ${VVRFRAMEWORK_DIR}/VVRScene/Release/VVRScene.lib 
    ${VVRFRAMEWORK_DIR}/GeoLib/Release/GeoLib.lib 
    ${VVRFRAMEWORK_DIR}/MathGeoLib/Release/MathGeoLib.lib
    )

    add_custom_command(TARGET ${SOLUTIONTITLE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${VVRFRAMEWORK_DIR}/VVRScene/Release" ${CMAKE_BINARY_DIR}/$<CONFIG>)
    add_custom_command(TARGET ${SOLUTIONTITLE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${VVRFRAMEWORK_DIR}/GeoLib/Release" ${CMAKE_BINARY_DIR}/$<CONFIG>)
    add_custom_command(TARGET ${SOLUTIONTITLE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${VVRFRAMEWORK_DIR}/MathGeoLib/Release" ${CMAKE_BINARY_DIR}/$<CONFIG>)
endif()

add_custom_command(TARGET ${SOLUTIONTITLE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${VVRFRAMEWORK_DIR}/VVRScene/lib" ${CMAKE_BINARY_DIR}/$<CONFIG>)
 
add_custom_command(TARGET ${SOLUTIONTITLE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/data" ${CMAKE_BINARY_DIR}/$<CONFIG>/data)
add_custom_command(TARGET ${SOLUTIONTITLE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/resources" ${CMAKE_BINARY_DIR}/$<CONFIG>/resources)
