#!/bin/bash
set -e
set -u

USERNAME=$(whoami)
docker run -it \
	--mount type=bind,source="$(pwd)",target=/home/${USERNAME}/robot_AQT \
	--mount type=bind,source="/data1/honda/robot_AQT_results",target=/home/${USERNAME}/robot_AQT/results \
	--user=$(id -u $USER):$(id -g $USER) \
	--env="DISPLAY" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--rm \
	--gpus=all \
	--workdir=/home/${USERNAME}/robot_AQT \
	--name=${USERNAME}_isaacgym_container_robot ${USERNAME}/isaacgym /bin/bash