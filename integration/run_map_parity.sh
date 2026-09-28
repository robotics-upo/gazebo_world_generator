#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 CLASSIC_PLUGIN_CHECKOUT" >&2
    exit 2
fi

CLASSIC_PLUGIN=$(realpath "$1")
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
ARTIFACTS=$(mktemp -d /tmp/gwg-map-parity.XXXXXX)
mkdir -p "$ARTIFACTS/classic" "$ARTIFACTS/harmonic"
echo "Map artifacts: $ARTIFACTS"

cd "$REPO_DIR"
docker build -f docker/Dockerfile.classic -t gwg-classic:ci .
docker build -f docker/Dockerfile.harmonic -t gwg-harmonic:ci .
docker build --build-context "plugin=$CLASSIC_PLUGIN" \
    -f docker/Dockerfile.classic-map -t gwg-classic-map:ci .
docker build -f docker/Dockerfile.harmonic-map -t gwg-harmonic-map:ci .

docker run --rm --shm-size=1g -v "$ARTIFACTS/classic:/maps" \
    gwg-classic-map:ci bash -lc \
    'source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && export PYTHONPATH="/ws/src/gazebo_world_generator:$PYTHONPATH" && python3 /ws/src/gazebo_world_generator/integration/generate_map_fixtures.py classic /maps'
docker run --rm --shm-size=1g -v "$ARTIFACTS/harmonic:/maps" \
    gwg-harmonic-map:ci bash -lc \
    'source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && export PYTHONPATH="/ws/src/gazebo_world_generator:$PYTHONPATH" && python3 /ws/src/gazebo_world_generator/integration/generate_map_fixtures.py harmonic /maps'

docker run --rm -v "$ARTIFACTS:/maps:ro" gwg-classic-map:ci bash -lc \
    'source /opt/ros/humble/setup.bash && cd /ws/src/gazebo_world_generator && export PYTHONPATH=".:$PYTHONPATH" && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
     GWG_CLASSIC_MAP=/maps/classic/map_room GWG_HARMONIC_MAP=/maps/harmonic/map_room \
     GWG_CLASSIC_CONNECTED_MAP=/maps/classic/connected GWG_HARMONIC_CONNECTED_MAP=/maps/harmonic/connected \
     GWG_CLASSIC_WAREHOUSE_MAP=/maps/classic/warehouse GWG_HARMONIC_WAREHOUSE_MAP=/maps/harmonic/warehouse \
     python3 -m pytest -q -rs tests/integration/test_map_parity.py'
