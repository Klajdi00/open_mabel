#!/bin/bash

# Define the permanent config folder
CONFIG_DIR=~/open_mabel/ros/src/mabel_sensors/config
mkdir -p $CONFIG_DIR

# Check if the GUI actually saved the file
if [ ! -f /tmp/calibrationdata.tar.gz ]; then
    echo "[ERROR] No calibration file found in /tmp/. Did you click SAVE in the GUI?"
    exit 1
fi

if [ "$1" == "head" ]; then
    tar -xzf /tmp/calibrationdata.tar.gz -C $CONFIG_DIR
    mv $CONFIG_DIR/left.yaml $CONFIG_DIR/head_left_cam.yaml
    mv $CONFIG_DIR/right.yaml $CONFIG_DIR/head_right_cam.yaml
    
    # Automatically rewrite the internal camera_name tags!
    sed -i 's/camera_name: .*/camera_name: head_left_cam/g' $CONFIG_DIR/head_left_cam.yaml
    sed -i 's/camera_name: .*/camera_name: head_right_cam/g' $CONFIG_DIR/head_right_cam.yaml
    echo "[SUCCESS] MABEL Head Stereo YAMLs saved and formatted in workspace!"

elif [ "$1" == "left_wrist" ]; then
    tar -xzf /tmp/calibrationdata.tar.gz -C $CONFIG_DIR
    mv $CONFIG_DIR/ost.yaml $CONFIG_DIR/left_wrist_cam.yaml
    
    # Automatically rewrite the internal camera_name tag!
    sed -i 's/camera_name: .*/camera_name: left_wrist_cam/g' $CONFIG_DIR/left_wrist_cam.yaml
    echo "[SUCCESS] Left Wrist YAML saved and formatted in workspace!"

elif [ "$1" == "right_wrist" ]; then
    tar -xzf /tmp/calibrationdata.tar.gz -C $CONFIG_DIR
    mv $CONFIG_DIR/ost.yaml $CONFIG_DIR/right_wrist_cam.yaml
    
    # Automatically rewrite the internal camera_name tag!
    sed -i 's/camera_name: .*/camera_name: right_wrist_cam/g' $CONFIG_DIR/right_wrist_cam.yaml
    echo "[SUCCESS] Right Wrist YAML saved and formatted in workspace!"

else
    echo "Usage: ./save_calib.sh [head | left_wrist | right_wrist]"
    exit 1
fi

# Clean up the temp folder
rm -f /tmp/calibrationdata.tar.gz
