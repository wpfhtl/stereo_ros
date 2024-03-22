cmake_minimum_required(VERSION 3.16)

# project(sgm LANGUAGES CXX CUDA)

# dependent packages
find_package(CUDA REQUIRED)

set(CUDA_ARCHS "52;61;72;75;86" CACHE STRING "List of architectures to generate device code for")

# if(BUILD_OPENCV_WRAPPER)
# 	find_package(OpenCV REQUIRED core)
# endif()

# library type
set(SGM_LIB_TYPE SHARED) # STATIC) #
if(LIBSGM_SHARED)
	set(SGM_LIB_TYPE SHARED)
endif()

file(GLOB CUDA_SOURCES ./*.cu)
cuda_add_library(cusgm ${SGM_LIB_TYPE} ${CUDA_SOURCES})

# target configuration
file(GLOB SRCS ./*.cpp ./*.cu ./*.h*)
message(${SRCS})
# target_sources(sgm PRIVATE ${SRCS})
# target_include_directories(sgm PRIVATE $<$<BOOL:${BUILD_OPENCV_WRAPPER}>:${OpenCV_INCLUDE_DIRS}>)
# target_compile_features(sgm PRIVATE cxx_std_11)
# target_link_libraries(sgm PUBLIC cudart $<$<BOOL:${BUILD_OPENCV_WRAPPER}>:${OpenCV_LIBS}>)
set_target_properties(cusgm PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHS})
target_compile_features(cusgm PRIVATE cxx_std_11)
# target_link_libraries(sgm PUBLIC cudart)
target_link_libraries(cusgm ${CUDA_CUDART_LIBRARY})
add_library(sgm ${SGM_LIB_TYPE} ${SRCS})
target_link_libraries(sgm cusgm)

# target_compile_options(sgm PRIVATE
# $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wall -O3>
# $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wall -O3>
# $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4819>
# $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
# )

# install(
# 	TARGETS sgm
# 	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
# 	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
# 	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
# )

# install(
# 	DIRECTORY ${LIBSGM_INCLUDE_DIR}
# 	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# 	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
# )

# install(
# 	FILES ${LIBSGM_ROOT_DIR}/FindLibSGM.cmake
# 	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
