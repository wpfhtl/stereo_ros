cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
project(stereo_ros)

find_package(catkin REQUIRED 
cv_bridge 
dynamic_reconfigure 
# image_geometry 
# image_proc 
image_transport 
message_filters nodelet sensor_msgs stereo_msgs pcl_ros pcl_conversions)
find_package(Boost REQUIRED COMPONENTS thread)
# find_package(PkgConfig)
# pkg_search_module(Eigen3 REQUIRED eigen3)
find_package(Eigen3 REQUIRED)
if (EIGEN3_FOUND)
    message(STATUS "found Eigen3:\"${EIGEN3_INCLUDE_DIR}\",ENABLE_EIGEN3 is true")
    include_directories(${EIGEN3_INCLUDE_DIR})
    add_definitions(-DENABLE_EIGEN3)
   list(APPEND LINK_LIB_LIST ${EIGEN3_LIBRARIES})
endif (EIGEN3_FOUND)


find_package(OpenCV  REQUIRED core imgproc calib3d highgui)
include_directories(${OpenCV_INCLUDE_DIRS})

if(cv_bridge_VERSION VERSION_GREATER "1.12.0")
  add_compile_options(-std=c++11)
endif()

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED COMPONENTS system filesystem program_options)
include_directories(${Boost_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
include_directories(${CUDA_INCLUDE_DIRS})

# include_directories("stereo_cuda")
# add_subdirectory(stereo_cuda) 

catkin_package(
  CATKIN_DEPENDS 
  cv_bridge 
  dynamic_reconfigure 
  # image_geometry 
  # image_proc 
  image_transport 
  message_filters nodelet eigen_conversions geometry_msgs sensor_msgs stereo_msgs pcl_ros pcl_conversions
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

include_directories(${Eigen_INCLUDE_DIRS})
link_libraries(${Boost_LIBRARIES} ${OpenCV_LIBRARIES})
include_directories(${catkin_INCLUDE_DIRS} include)
add_definitions(-DOPENCV_TRAITS_ENABLE_DEPRECATED)

add_executable(stereo_ros src/stereo_ros.cpp)
target_link_libraries(stereo_ros ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
install(TARGETS stereo_ros
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

