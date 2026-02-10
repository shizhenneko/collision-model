cmake_minimum_required(VERSION 3.0.2)
project(collision_model)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf
  cv_bridge
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/inference_node.py
  scripts/navigation_node.py
  scripts/onnx_infer_demo.py
  scripts/send_goal.py
  scripts/build_dataset.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch checkpoints
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
