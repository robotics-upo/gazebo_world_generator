# Gazebo World Generator
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![ROS Version](https://img.shields.io/badge/ROS%202-Humble%20%7C%20Jazzy-orange.svg)
![Python Version](https://img.shields.io/badge/Python-3.10+-green.svg)

**Build Complex Gazebo Worlds with a Single Sentence.**

Gazebo World Generator turns plain English descriptions into Gazebo worlds, places furniture, and can produce 2D navigation maps. It supports Gazebo Classic on ROS 2 Humble and Gazebo Harmonic on ROS 2 Jazzy.

Console Interface             |  Generated World
:-------------------------:|:-------------------------:
![console](https://github.com/robotics-upo/gazebo_world_generator/blob/master/media/gazebo_world_gen_console.gif) | *"A 12x10 m warehouse with 8 pallets and 10 shelves connected to a small office with a desk and a chair"* ![world](https://github.com/robotics-upo/gazebo_world_generator/blob/master/media/gazebo_world_gen_world.gif)

> [!NOTE]
> This is a work in progress. Generated worlds may need manual adjustments.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## Features

- **Natural language input** – describe the world in plain English
- **Automatic placement** – furniture grouping for offices, grid layouts for warehouses
- **Navigation maps** – 2D occupancy maps when the map plugin is available
- **Interactive refinement** – modify existing worlds with simple commands
- **Room dimensions** – specify exact room sizes

## Installation

All options need an OpenAI-compatible LLM server (LM Studio, Ollama, vLLM, OpenAI API, ...).

| Option | Provides | Requires |
| --- | --- | --- |
| [Native ROS 2](#native-ros-2) | Worlds, maps, `ros2 run` | ROS 2 Humble + Gazebo Classic 11, or ROS 2 Jazzy + Gazebo Harmonic |
| [Docker](#docker) | Worlds and maps without ROS on the host | Docker with Compose |
| [Python only](#python-only) | Worlds only (no maps) | Python 3.10+ |

### Native ROS 2

```bash
cd ~/ros2_ws/src
git clone https://github.com/robotics-upo/gazebo_world_generator.git
pip install --user -r gazebo_world_generator/requirements.txt

# Humble / Classic
sudo apt-get install ros-humble-gazebo-ros-pkgs
# Jazzy / Harmonic
sudo apt-get install ros-jazzy-ros-gz

cd ~/ros2_ws
colcon build --packages-select gazebo_world_generator
source install/setup.bash
```

On Ubuntu 24.04 (Jazzy), pip refuses to install into the system Python
(`externally-managed-environment`); add `--break-system-packages` to the pip
command.

**Map plugin (optional).** Maps need `gazebo_ros2_2dmap_plugin` in a sourced
workspace. `ros2 pkg executables gazebo_ros2_2dmap_plugin` lists
`generate_map.sh` when it is available. Without it, worlds are still generated
and the map step is reported as skipped.

```bash
# Humble: companion plugin, humble branch
sudo apt-get install ros-humble-nav2-map-server
git clone -b humble https://github.com/robotics-upo/gazebo_ros2_2Dmap_plugin.git \
    ~/ros2_ws/src/gazebo_ros2_2dmap_plugin

# Jazzy: Harmonic port shipped in this repository
cp -a ~/ros2_ws/src/gazebo_world_generator/integration/harmonic_map_plugin \
    ~/ros2_ws/src/gazebo_ros2_2dmap_plugin

cd ~/ros2_ws && colcon build --packages-select gazebo_ros2_2dmap_plugin
source install/setup.bash
```

### Docker

The images bundle ROS 2, the simulator, the map plugin and the generator:
`classic` (Humble + Gazebo Classic) and `harmonic` (Jazzy + Gazebo Harmonic).
From a clone of this repository:

```bash
mkdir -p generated_worlds ~/.gazebo/models   # create before the first run
docker compose build classic
docker compose run --rm classic              # interactive menu
docker compose run --rm classic --description "8m x 6m office with 1 desk and 1 office chair"
```

Use `harmonic` instead of `classic` for Gazebo Harmonic. Outputs appear in
`./generated_worlds` (printed as `/data/generated_worlds` inside the
container). Downloaded models go to `~/.gazebo/models`, so the host simulator
can open the worlds (see [Opening worlds](#opening-worlds)).

The container uses host networking, so an LLM server on `localhost` works.
Set these in your shell or in a `.env` file next to `compose.yaml`:

| Variable | Purpose |
| --- | --- |
| `GAZEBO_WORLD_GEN_LLM__SERVER_URL`, `GAZEBO_WORLD_GEN_LLM__MODEL_NAME` | LLM endpoint and model |
| `OPENAI_API_KEY`, `GITHUB_TOKEN` | Optional credentials |
| `GWG_OUTPUT_DIR`, `GWG_MODELS_DIR` | Host output and model directories (defaults above) |
| `GWG_UID`, `GWG_GID` | Container user, default `1000`; set to `$(id -u)`/`$(id -g)` if yours differ |

The container reads neither your personal config file nor custom model
directories; pass settings as `GAZEBO_WORLD_GEN_...` variables and mount extra
model directories yourself.

Experimental: open a world inside the container over X11 (may need GPU setup):

```bash
xhost +local:
docker compose run --rm classic-gui gazebo generated_worlds/worlds/<name>.sdf
docker compose run --rm harmonic-gui gz sim generated_worlds/worlds/<name>.sdf
```

### Python only

```bash
pipx install git+https://github.com/robotics-upo/gazebo_world_generator.git
generate_world --simulator harmonic --description "Office with 4 desks and chairs"
```

`pip install .` in a virtual environment also works. Without ROS the simulator
defaults to `classic`, and no map is generated.

## Usage

```bash
ros2 run gazebo_world_generator generate_world
```

With no arguments, an interactive menu opens: generate a world, refine an
existing world, test the LLM connection, or exit. Pass `--description` to
generate directly:

```bash
ros2 run gazebo_world_generator generate_world \
    --description "12m x 10m warehouse with 8 storage racks and 5 pallets"
```

| Option | Meaning |
| --- | --- |
| `--description` | World description; omit for interactive mode |
| `--output` | Exact SDF path. Default: timestamped file in `generated_worlds/worlds/` |
| `--llm-server`, `--model` | Override the configured LLM endpoint and model |
| `--simulator` | `classic` or `harmonic`; default follows `ROS_DISTRO` (Jazzy → Harmonic, otherwise Classic) |
| `--seed` | Seed for placement randomness (LLM output may still vary) |
| `--debug` | Verbose logging in the log file |

**Writing descriptions.** Give counts; unspecified furniture is not added
("warehouse" is an empty room). Room sizes are recognised as
`"12m x 10m office"`, `"warehouse (20x15m)"` or `"office of size 8x6"`.
Warehouses with racks and pallets use grid placement. Check the reported object
count: the LLM can misread a request, and generation fails with the object type
named if no suitable model is found.

**Refinement.** Choose menu option 2 and describe changes, e.g. "Add 3 more
desks with chairs", "Remove all bookshelves", "Move desks closer to the entrance".

### Output files

```plaintext
generated_worlds/
├── worlds/world_<timestamp>.sdf           # Gazebo world
├── logs/world_<timestamp>.log             # Detailed generation log
└── occupancy_maps/world_<timestamp>.{pgm,yaml}   # 2D map, when enabled and available
```

Logs and maps always go under the output directory, even with `--output`.

### Opening worlds

Downloaded models live in `~/.gazebo/models`; expose that directory to the
simulator (plus any custom `models.search_paths`):

```bash
# Humble / Classic
export GAZEBO_MODEL_PATH="$HOME/.gazebo/models${GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}"
gazebo generated_worlds/worlds/<name>.sdf

# Jazzy / Harmonic
export GZ_SIM_RESOURCE_PATH="$HOME/.gazebo/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
gz sim generated_worlds/worlds/<name>.sdf
```

On Classic, close Gazebo before generating a map: the map script stops running
Gazebo servers, so the generator reports an error instead of doing so.

## Configuration

Defaults live in `config/generator_config.yaml`:

```yaml
llm:
  server_url: "http://localhost:1234/v1"
  model_name: "your-model-name"
  context_window: 8192       # tokens; raise it if your model allows
output:
  base_directory: "generated_worlds"
  auto_generate_map: true
placement:
  max_design_rounds: 3       # LLM layout proposals/revisions per room
  review_rounds: 1           # LLM reviews of each finished room (0 = off)
  corridor_width: 1.2        # meters
```

Settings are merged from lowest to highest priority:

1. `config/generator_config.yaml`
2. `~/.config/gazebo_world_generator/generator_config.yaml` (only the keys you override)
3. `GAZEBO_WORLD_GEN_...` environment variables, nested with `__`
   (e.g. `GAZEBO_WORLD_GEN_OUTPUT__AUTO_GENERATE_MAP=false`)
4. Command-line options

Keep machine-specific settings such as your LLM endpoint in the personal file
rather than the repository. If a repository change seems ignored, check the
personal file.

## How It Works

1. **Parsing** – explicit room sizes are extracted, then the LLM turns the description into rooms with object types and counts.
2. **Room layout** – rooms, corridors and doorways are positioned.
3. **Model resolution** – each object type is matched to a local or online (Gazebo Fuel, GitHub) model whose meshes, textures and measured size are checked.
4. **Placement** – the LLM designs each room's layout and names the requirements it should meet (e.g. a 0.8 m route between doorways). The generator measures the layout against them and sends the problems back, with a top-down image when the model accepts images, for up to `max_design_rounds` revisions. Any overlaps left are then pushed apart.
5. **Output** – the SDF world is written, then the occupancy map when enabled.

To look at a world without opening the GUI, render it from above (one image
per room plus an overview):

```bash
ros2 run gazebo_world_generator render_world generated_worlds/worlds/<name>.sdf --out renders/
```

## Troubleshooting

**LLM connection.** Use menu option 3, or `curl <server_url>/models`.

**No suitable model found.** The generator searches `models.search_paths`, the
model cache, `GAZEBO_MODEL_PATH`/`GZ_SIM_RESOURCE_PATH` and online providers,
including synonyms (e.g. "couch" for "sofa"). Models without visible geometry,
with missing mesh or texture files, or too small for the requested furniture
(e.g. a `DeskPortrait` for "desk") are rejected; the world log lists why. The
chosen models must also be visible to the simulator when opening the world.

**Context limit errors** (`'max_tokens' is too large`). Output tokens are
reduced automatically to fit the model. For very large worlds, split the
request into smaller rooms or use a model with a larger context.

**Object overlaps.** Enlarge the room or reduce the object count; see
`generated_worlds/logs/` for warnings.

**Build or package not found.**

```bash
cd ~/ros2_ws && rm -rf build/ install/ log/
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select gazebo_world_generator
source install/setup.bash
```

## Development

```bash
pip install -r requirements-dev.txt
python3 -m pytest -q tests                              # or: colcon test --packages-select gazebo_world_generator
docker build -f docker/Dockerfile.classic .             # CI test images; the suite runs during the build
docker build -f docker/Dockerfile.harmonic .
bash integration/run_map_parity.sh /path/to/humble-plugin   # Classic vs Harmonic map comparison
```

The Harmonic map plugin in `integration/harmonic_map_plugin` is a standalone
ROS package ported from the companion plugin's Fortress branch. CI builds both
plugins, validates generated SDF files, and compares Classic and Harmonic maps
for three layouts. The parity runner prints the `/tmp/gwg-map-parity.*`
directory holding its artifacts. Standalone `gz sdf -k` checks of worlds with
`model://` includes may need `SDF_PATH` set to the model directory.

## License

MIT License - Copyright (c) 2026 Service Robotics Lab. See [LICENSE](LICENSE).
