cmake_minimum_required(VERSION 3.10)
project(mabel_gateway)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
