cmake_minimum_required(VERSION 3.10)
project(mabel_sensors)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# ---------------------------------------------------
# INSTALL RULES
# ---------------------------------------------------
# This tells the compiler to copy your launch folder
# into the ROS 2 installation directory.
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Camera intrinsics / RealSense parameter YAMLs (the launch files read these via
# get_package_share_directory('mabel_sensors')/config).
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# info_injector.py (head-stereo camera_info bridge) is run by sensors.launch.py
# via its share path -- install scripts/ so that path resolves on the robot.
install(DIRECTORY scripts
  DESTINATION share/${PROJECT_NAME}
)

# ---------------------------------------------------
# TESTING (Default ROS 2 boilerplate)
# ---------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# This must be the very last line in the file!
ament_package()