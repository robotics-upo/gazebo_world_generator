# User-facing runtime image. Build with --build-arg ROS_DISTRO=humble
# (Gazebo Classic) or ROS_DISTRO=jazzy (Gazebo Harmonic). The CI test images
# are Dockerfile.classic and Dockerfile.harmonic.
ARG ROS_DISTRO=humble

FROM ros:humble-ros-base AS base-humble
ARG CLASSIC_PLUGIN_REF=74a8939e09f0aeeec63e221875429b2ea5e45f7d
RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3-pip python3-colcon-common-extensions \
        ros-humble-gazebo-ros-pkgs ros-humble-nav2-map-server libsdformat-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/robotics-upo/gazebo_ros2_2Dmap_plugin.git \
        /ws/src/gazebo_ros2_2dmap_plugin \
    && git -C /ws/src/gazebo_ros2_2dmap_plugin checkout --quiet "$CLASSIC_PLUGIN_REF" \
    && rm -rf /ws/src/gazebo_ros2_2dmap_plugin/.git
ENV GAZEBO_WORLD_GEN_MODELS__SEARCH_PATHS='["/data/models", "/usr/share/gazebo-11/models"]' \
    GAZEBO_MODEL_PATH=/data/models

FROM ros:jazzy-ros-base AS base-jazzy
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-colcon-common-extensions ros-jazzy-ros-gz \
    && rm -rf /var/lib/apt/lists/*
COPY integration/harmonic_map_plugin /ws/src/gazebo_ros2_2dmap_plugin
ENV GAZEBO_WORLD_GEN_MODELS__SEARCH_PATHS='["/data/models"]' \
    GZ_SIM_RESOURCE_PATH=/data/models \
    PIP_BREAK_SYSTEM_PACKAGES=1

FROM base-${ROS_DISTRO}
SHELL ["/bin/bash", "-c"]
# Software rendering for render_world when no display or GPU is available.
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb xauth libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY . /ws/src/gazebo_world_generator
WORKDIR /ws
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build --packages-select gazebo_ros2_2dmap_plugin gazebo_world_generator \
    && rm -rf build log
COPY docker/entrypoint.sh /entrypoint.sh

# Any UID may run the container, so HOME must be writable by everyone.
RUN mkdir -p -m 1777 /home/gwg /data
ENV HOME=/home/gwg \
    GAZEBO_WORLD_GEN_OUTPUT__BASE_DIRECTORY=/data/generated_worlds \
    GAZEBO_WORLD_GEN_MODELS__CACHE_DIRECTORY=/data/models
WORKDIR /data
ENTRYPOINT ["/entrypoint.sh"]
