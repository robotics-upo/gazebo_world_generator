#!/bin/bash
# Script to generate and save a 2D occupancy map from a Gazebo Harmonic world file
#
# Usage: ./generate_map.sh <world_file> [output_directory]
#
# Example: ./generate_map.sh ~/my_world.sdf ./maps
#
# The output map will have the same name as the world file (without extension)
# 
# This script uses Gazebo Harmonic for map generation.

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <world_file> [output_directory]"
    echo "Example: $0 ~/my_world.sdf ./maps"
    echo ""
    echo "The output map will automatically be named after the world file."
    echo "For example: my_world.sdf -> my_world.yaml + my_world.pgm"
    exit 1
fi

WORLD_FILE="$1"
OUTPUT_DIR="${2:-.}"  # Default to current directory if not specified

# Extract map name from world file (remove path and extension)
MAP_NAME=$(basename "$WORLD_FILE" | sed 's/\.[^.]*$//')

# Check if world file exists
if [ ! -f "$WORLD_FILE" ]; then
    echo "Error: World file '$WORLD_FILE' not found!"
    exit 1
fi

# Validate the world file has proper SDF structure
if ! grep -q '<sdf\|<world' "$WORLD_FILE" 2>/dev/null; then
    echo "Error: World file does not appear to be a valid SDF/world file!"
    echo "Expected to find <sdf> or <world> tags."
    exit 1
fi

# Set up model resource paths for Ignition Gazebo
# This helps find models referenced as model://name
if [ -d "$HOME/.gazebo/models" ]; then
    export IGN_GAZEBO_RESOURCE_PATH="${IGN_GAZEBO_RESOURCE_PATH:+$IGN_GAZEBO_RESOURCE_PATH:}$HOME/.gazebo/models"
    export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:+$GZ_SIM_RESOURCE_PATH:}$HOME/.gazebo/models"
fi
if [ -d "$HOME/.ignition/gazebo_models" ]; then
    export IGN_GAZEBO_RESOURCE_PATH="${IGN_GAZEBO_RESOURCE_PATH:+$IGN_GAZEBO_RESOURCE_PATH:}$HOME/.ignition/gazebo_models"
    export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:+$GZ_SIM_RESOURCE_PATH:}$HOME/.ignition/gazebo_models"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Detect which Gazebo command is available
# Ignition Fortress uses "ign gazebo", newer versions use "gz sim"
if command -v gz &> /dev/null && gz sim --help &> /dev/null; then
    GZ_CMD="gz sim"
    GZ_SERVICE_CMD="gz service"
    GZ_TOPIC_CMD="gz topic"
    echo "Detected Gazebo Sim (Harmonic)"
elif command -v ign &> /dev/null; then
    GZ_CMD="ign gazebo"
    GZ_SERVICE_CMD="ign service"
    GZ_TOPIC_CMD="ign topic"
    echo "Detected Ignition Gazebo (Fortress)"
else
    echo "Error: Neither 'ign gazebo' nor 'gz sim' found!"
    echo "Please install Ignition Gazebo Fortress: https://gazebosim.org/docs/fortress/install"
    exit 1
fi

# Automatically find and configure the plugin library path
PLUGIN_PATH=""

# Start with existing path if set
if [ -n "$IGN_GAZEBO_SYSTEM_PLUGIN_PATH" ]; then
    PLUGIN_PATH="$IGN_GAZEBO_SYSTEM_PLUGIN_PATH"
fi

# Try to find the plugin via ros2 pkg prefix (works when workspace is sourced)
PKG_PREFIX=$(ros2 pkg prefix gazebo_ros2_2dmap_plugin 2>/dev/null || echo "")
if [ -n "$PKG_PREFIX" ] && [ -f "$PKG_PREFIX/lib/libgz_2Dmap_system.so" ]; then
    if [ -n "$PLUGIN_PATH" ]; then
        PLUGIN_PATH="$PKG_PREFIX/lib:$PLUGIN_PATH"
    else
        PLUGIN_PATH="$PKG_PREFIX/lib"
    fi
    echo "✓ Found plugin at: $PKG_PREFIX/lib/libgz_2Dmap_system.so"
else
    # Fallback: try common workspace locations
    COMMON_PATHS=(
        "$HOME/ros2_ws/install/gazebo_ros2_2dmap_plugin/lib"
        "$HOME/colcon_ws/install/gazebo_ros2_2dmap_plugin/lib"
        "$(dirname "$0")/../lib"
    )
    for path in "${COMMON_PATHS[@]}"; do
        if [ -f "$path/libgz_2Dmap_system.so" ]; then
            if [ -n "$PLUGIN_PATH" ]; then
                PLUGIN_PATH="$path:$PLUGIN_PATH"
            else
                PLUGIN_PATH="$path"
            fi
            echo "✓ Found plugin at: $path/libgz_2Dmap_system.so"
            break
        fi
    done
fi

# Check if plugin was found
if [ -z "$PLUGIN_PATH" ] || ! echo "$PLUGIN_PATH" | grep -q "libgz_2Dmap_system"; then
    # Check if the file exists in any of the paths
    FOUND=false
    for p in $(echo "$PLUGIN_PATH" | tr ':' ' '); do
        if [ -f "$p/libgz_2Dmap_system.so" ]; then
            FOUND=true
            break
        fi
    done
    if [ "$FOUND" = false ]; then
        echo "WARNING: Could not automatically find libgz_2Dmap_system.so"
        echo "Make sure your workspace is sourced: source ~/ros2_ws/install/setup.bash"
    fi
fi

# Export the plugin path for Ignition Gazebo (both naming conventions)
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH="$PLUGIN_PATH"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$PLUGIN_PATH"

# Create a temporary world file with the plugin injected
TEMP_WORLD_FILE=""
LOG_FILE=$(mktemp /tmp/gazebo_map_gen.XXXXXX.log)
PLUGIN_INJECTED=false

# Check if the world file already has the plugin
if grep -q "gz_2Dmap_system\|OccupancyMapFromWorld" "$WORLD_FILE"; then
    echo "Plugin already present in world file"
    TEMP_WORLD_FILE="$WORLD_FILE"
else
    echo "Injecting occupancy map plugin into world file..."
    TEMP_WORLD_FILE=$(mktemp /tmp/gazebo_map_gen_world.XXXXXX.sdf)
    
    # Get absolute output path for the map files
    OUTPUT_PATH_ABS="$(cd "$OUTPUT_DIR" && pwd)/$MAP_NAME"
    
    # Ignition Fortress-style plugin configuration
    # Match the Classic map resolution and slicing height.
    PLUGIN_XML="    <plugin filename=\"gz_2Dmap_system\" name=\"gz::sim::systems::OccupancyMapFromWorld\">
      <map_resolution>0.05</map_resolution>
      <map_height>0.2</map_height>
      <init_robot_x>0</init_robot_x>
      <init_robot_y>0</init_robot_y>
      <output_path>$OUTPUT_PATH_ABS</output_path>
    </plugin>"
    
    # Find the last </world> tag and insert plugin before it
    awk -v plugin="$PLUGIN_XML" '
        /<\/world>/ && !done {
            print plugin
            done=1
        }
        { print }
    ' "$WORLD_FILE" > "$TEMP_WORLD_FILE"
    
    PLUGIN_INJECTED=true
    echo "✓ Plugin injected into temporary world file"
fi

echo "================================================"
echo "Automated Gazebo Harmonic 2D Map Generator"
echo "================================================"
echo "Processing: $(basename "$WORLD_FILE")"
echo "Output dir: $OUTPUT_DIR"
echo "================================================"

# PIDs to track for cleanup
GAZEBO_PID=""

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    
    # Stop only the simulator started by this invocation.
    if [ -n "$GAZEBO_PID" ]; then
        kill "$GAZEBO_PID" 2>/dev/null || true
        wait "$GAZEBO_PID" 2>/dev/null || true
    fi
    
    # Remove temporary world file if it was created
    if [ "$PLUGIN_INJECTED" = true ] && [ -f "$TEMP_WORLD_FILE" ]; then
        rm -f "$TEMP_WORLD_FILE" 2>/dev/null || true
    fi
    
    # Remove temporary log file
    rm -f "$LOG_FILE" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo "[1/4] Starting Gazebo Harmonic server..."

# Plugin path was already exported above, just show it for debugging
if [ -n "$PLUGIN_PATH" ]; then
    echo "Using plugin path: $PLUGIN_PATH"
fi

# Re-export resource paths just before starting Gazebo to ensure they're set
if [ -d "$HOME/.ignition/gazebo_models" ]; then
    export IGN_GAZEBO_RESOURCE_PATH="$HOME/.ignition/gazebo_models${IGN_GAZEBO_RESOURCE_PATH:+:$IGN_GAZEBO_RESOURCE_PATH}"
fi
if [ -d "$HOME/.gazebo/models" ]; then
    export IGN_GAZEBO_RESOURCE_PATH="$HOME/.gazebo/models${IGN_GAZEBO_RESOURCE_PATH:+:$IGN_GAZEBO_RESOURCE_PATH}"
fi
if [ -n "$IGN_GAZEBO_RESOURCE_PATH" ]; then
    echo "Using model path: $IGN_GAZEBO_RESOURCE_PATH"
fi

# Start Ignition Gazebo in server-only mode (no GUI)
# -r: run simulation immediately (needed for PostUpdate to be called)
$GZ_CMD -s -r "$TEMP_WORLD_FILE" -v 3 > "$LOG_FILE" 2>&1 &
GAZEBO_PID=$!

# Wait for Gazebo to start
echo "[2/4] Waiting for Gazebo initialization..."
sleep 5

# Check if Gazebo started successfully
if ! ps -p $GAZEBO_PID > /dev/null; then
    echo "ERROR: Gazebo Harmonic failed to start!"
    echo ""
    echo "=== Log output ==="
    tail -40 "$LOG_FILE" 2>/dev/null
    echo ""
    echo "=== Troubleshooting ==="
    
    # Check for common issues
    if grep -q "Unable to find uri\|Unable to find file" "$LOG_FILE" 2>/dev/null; then
        echo "⚠️  MISSING MODELS: The world file references models that aren't installed."
        echo "   Try downloading them from Fuel: https://app.gazebosim.org/fuel"
        echo "   Or set IGN_GAZEBO_RESOURCE_PATH to include your model directories."
    fi
    
    if grep -q "plugin.*not found\|Failed to load plugin" "$LOG_FILE" 2>/dev/null; then
        echo "⚠️  MISSING PLUGINS: The world file references plugins that aren't installed."
        echo "   This may happen if the world was created for a different Gazebo version."
    fi
    
    exit 1
fi

echo "[3/4] Triggering map generation..."

# Call the Gazebo service to generate the map
# The service is advertised by the plugin
# Note: --req '' is required even for Empty message type
if $GZ_SERVICE_CMD -l 2>/dev/null | grep -q "generate_map"; then
    $GZ_SERVICE_CMD -s /gazebo_2Dmap_plugin/generate_map \
        --reqtype gz.msgs.Empty --reptype gz.msgs.Empty \
        --timeout 5000 --req '' > /dev/null 2>&1 || true
    sleep 2
else
    echo "WARNING: generate_map service not found yet, waiting..."
    sleep 3
    $GZ_SERVICE_CMD -s /gazebo_2Dmap_plugin/generate_map \
        --reqtype gz.msgs.Empty --reptype gz.msgs.Empty \
        --timeout 5000 --req '' > /dev/null 2>&1 || true
    sleep 2
fi

echo "[4/4] Verifying map files..."
cd "$OUTPUT_DIR"

# Wait for map files to be created (poll with timeout)
# Large worlds may take longer to generate
MAX_WAIT=30
for i in $(seq 1 $MAX_WAIT); do
    if [ -f "${MAP_NAME}.yaml" ] && [ -f "${MAP_NAME}.pgm" ]; then
        break
    fi
    sleep 1
done

# Verify files were created
if [ -f "${MAP_NAME}.yaml" ] && [ -f "${MAP_NAME}.pgm" ]; then
    echo ""
    echo "================================================"
    echo "✓ SUCCESS: Map generated successfully"
    echo "================================================"
    echo "World:  $(basename "$WORLD_FILE")"
    echo "Output: $OUTPUT_DIR/$MAP_NAME.yaml"
    echo "        $OUTPUT_DIR/$MAP_NAME.pgm"
    echo "================================================"
    
    # Success - exit cleanly (cleanup trap will run automatically)
    exit 0
else
    echo "================================================"
    echo "ERROR: Map files were not created!"
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Check if the plugin loaded: grep 'gz_2Dmap_plugin' /tmp/gazebo_map_gen.log"
    echo "2. Check for errors: cat /tmp/gazebo_map_gen.log"
    echo "3. Verify ros_gz_bridge is installed: ros2 pkg list | grep ros_gz"
    echo "4. Try running manually:"
    echo "   export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=$PLUGIN_PATH"
    echo "   $GZ_CMD -s $WORLD_FILE"
    echo "   ros2 run ros_gz_bridge parameter_bridge /map2d@nav_msgs/msg/OccupancyGrid[ignition.msgs.OccupancyGrid"
    echo "   $GZ_SERVICE_CMD -s /gazebo_2Dmap_plugin/generate_map --reqtype ignition.msgs.Empty --reptype ignition.msgs.Empty"
    echo "================================================"
    exit 1
fi
