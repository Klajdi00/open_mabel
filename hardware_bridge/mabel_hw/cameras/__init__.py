"""cameras — native-SDK camera ownership for the MABEL hardware bridge.

The bridge opens the camera SDKs directly (pyrealsense2 for the D435 front camera,
pyzed for the ZED-M head camera) and republishes the EXISTING ROS camera contract
(/camera/front/*, /camera/zed/*) plus the static mount TFs, so the SLAM / nvblox stack
runs unchanged with no realsense2_camera / zed_wrapper drivers. Each camera degrades
gracefully (absent -> retry, never fatal).
"""
