# Gazebo Harmonic 2D map plugin

This is the Gazebo Harmonic / ROS 2 Jazzy port of
[`robotics-upo/gazebo_ros2_2Dmap_plugin`](https://github.com/robotics-upo/gazebo_ros2_2Dmap_plugin),
based on its Fortress implementation. It lives here as a standalone ROS package
so a Jazzy workspace can build it directly, with no source patch at build time.
The package name and `generate_map.sh` command match the Classic companion
plugin. The script produces a PGM image and YAML metadata with the input
world's basename.

From a Jazzy workspace, copy this directory to `src/gazebo_ros2_2dmap_plugin`,
then run:

```bash
colcon build --packages-select gazebo_ros2_2dmap_plugin
source install/setup.bash
ros2 run gazebo_ros2_2dmap_plugin generate_map.sh world.sdf maps/
```

Set `GZ_SIM_RESOURCE_PATH` to directories containing any `model://` resources
used by the world. Map resolution defaults to 0.05 m and the occupancy slice
height is 0.2 m. The map generation script injects its system plugin into a
temporary copy of the world and starts Gazebo headlessly.

The original project is MIT licensed. The port's C++ files also carry
Apache-2.0 headers from their upstream components; both notices are included
in this directory.
