cmake_minimum_required(VERSION 3.10)
project(gazebo_ros2_2dmap_plugin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(gz-math7 REQUIRED)
find_package(gz-common5 REQUIRED)

add_library(gz_2Dmap_system SHARED src/gz_2Dmap_system.cc)
target_include_directories(gz_2Dmap_system PRIVATE include)
target_compile_options(gz_2Dmap_system PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(gz_2Dmap_system
  gz-sim8::gz-sim8
  gz-plugin2::gz-plugin2
  gz-transport13::gz-transport13
  gz-msgs10::gz-msgs10
  gz-math7::gz-math7
  gz-common5::gz-common5
)

install(TARGETS gz_2Dmap_system LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.hh")
install(PROGRAMS scripts/generate_map.sh DESTINATION lib/${PROJECT_NAME})

ament_package()
