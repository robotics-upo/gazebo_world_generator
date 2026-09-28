#!/bin/bash
# Source ROS and the workspace, then run the generator. Arguments starting
# with "-" (or none) go to generate_world; anything else runs as a command,
# e.g. `bash` or `gazebo world.sdf`.
set -e
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /ws/install/setup.bash
# Classic's resource and model paths (keeps /data/models from the image).
if [ -f /usr/share/gazebo/setup.sh ]; then
    source /usr/share/gazebo/setup.sh
fi
if [ "$#" -eq 0 ] || [ "${1#-}" != "$1" ]; then
    set -- ros2 run gazebo_world_generator generate_world "$@"
fi
exec "$@"
