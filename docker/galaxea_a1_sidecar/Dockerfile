FROM ros:noetic-ros-base-focal

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# The vendor SDK is never copied into this image. Operators mount a separately
# obtained, already-built A1_SDK tree read-only at run time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libconsole-bridge0.4 \
    libassimp5 \
    liboctomap-dev \
    liborocos-kdl1.4 \
    python3-serial \
    ros-noetic-controller-interface \
    ros-noetic-controller-manager \
    ros-noetic-joint-state-controller \
    ros-noetic-interactive-markers \
    ros-noetic-kdl-parser \
    ros-noetic-pluginlib \
    ros-noetic-robot-state-publisher \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-tf \
    ros-noetic-tf-conversions \
    ros-noetic-tf2-ros \
    ros-noetic-trac-ik-lib \
    ros-noetic-urdf \
    ros-noetic-urdf-parser-plugin \
    ros-noetic-xacro \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_MASTER_URI=http://127.0.0.1:11311
ENV ROS_HOSTNAME=127.0.0.1

COPY entrypoint.sh /usr/local/bin/openral-a1-entrypoint
RUN chmod 0755 /usr/local/bin/openral-a1-entrypoint

ENTRYPOINT ["/usr/local/bin/openral-a1-entrypoint"]
CMD ["bash"]
