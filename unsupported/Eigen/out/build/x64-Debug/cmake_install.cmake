# Install script for directory: E:/Hutch++/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "E:/Hutch++/unsupported/Eigen/out/install/x64-Debug")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/unsupported/Eigen/AdolcForward;/unsupported/Eigen/AlignedVector3;/unsupported/Eigen/ArpackSupport;/unsupported/Eigen/AutoDiff;/unsupported/Eigen/BVH;/unsupported/Eigen/EulerAngles;/unsupported/Eigen/FFT;/unsupported/Eigen/IterativeSolvers;/unsupported/Eigen/KroneckerProduct;/unsupported/Eigen/LevenbergMarquardt;/unsupported/Eigen/MatrixFunctions;/unsupported/Eigen/MoreVectorization;/unsupported/Eigen/MPRealSupport;/unsupported/Eigen/NonLinearOptimization;/unsupported/Eigen/NumericalDiff;/unsupported/Eigen/OpenGLSupport;/unsupported/Eigen/Polynomials;/unsupported/Eigen/Skyline;/unsupported/Eigen/SparseExtra;/unsupported/Eigen/SpecialFunctions;/unsupported/Eigen/Splines")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/unsupported/Eigen" TYPE FILE FILES
    "E:/Hutch++/unsupported/Eigen/AdolcForward"
    "E:/Hutch++/unsupported/Eigen/AlignedVector3"
    "E:/Hutch++/unsupported/Eigen/ArpackSupport"
    "E:/Hutch++/unsupported/Eigen/AutoDiff"
    "E:/Hutch++/unsupported/Eigen/BVH"
    "E:/Hutch++/unsupported/Eigen/EulerAngles"
    "E:/Hutch++/unsupported/Eigen/FFT"
    "E:/Hutch++/unsupported/Eigen/IterativeSolvers"
    "E:/Hutch++/unsupported/Eigen/KroneckerProduct"
    "E:/Hutch++/unsupported/Eigen/LevenbergMarquardt"
    "E:/Hutch++/unsupported/Eigen/MatrixFunctions"
    "E:/Hutch++/unsupported/Eigen/MoreVectorization"
    "E:/Hutch++/unsupported/Eigen/MPRealSupport"
    "E:/Hutch++/unsupported/Eigen/NonLinearOptimization"
    "E:/Hutch++/unsupported/Eigen/NumericalDiff"
    "E:/Hutch++/unsupported/Eigen/OpenGLSupport"
    "E:/Hutch++/unsupported/Eigen/Polynomials"
    "E:/Hutch++/unsupported/Eigen/Skyline"
    "E:/Hutch++/unsupported/Eigen/SparseExtra"
    "E:/Hutch++/unsupported/Eigen/SpecialFunctions"
    "E:/Hutch++/unsupported/Eigen/Splines"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/unsupported/Eigen/src")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/unsupported/Eigen" TYPE DIRECTORY FILES "E:/Hutch++/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("E:/Hutch++/unsupported/Eigen/out/build/x64-Debug/CXX11/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "E:/Hutch++/unsupported/Eigen/out/build/x64-Debug/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
