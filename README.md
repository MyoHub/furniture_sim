# furniture_sim

MuJoCo simulation assets for common household and laboratory furniture, packaged for [MyoSuite](https://github.com/MyoHub/myosuite).

## Installation

```bash
pip install furniture-sim
```

## Usage

```python
import mujoco
from furniture_sim import get_asset_root, resolve_asset

xml_path = resolve_asset("kitchen/kitchen1.xml")
model = mujoco.MjModel.from_xml_path(str(xml_path))
```

## Included Models

| `ventionTable.xml` | `simpleTable.xml` | `studyTable.xml` |
| :---: | :---: | :---: |
| ![ventionTable](https://raw.githubusercontent.com/MyoHub/furniture_sim/master/ventionTable/ventionTable.png "vention table") | ![simpleTable](https://raw.githubusercontent.com/MyoHub/furniture_sim/master/simpleTable/simpleTable.png "simple table") | ![studyTable](https://raw.githubusercontent.com/MyoHub/furniture_sim/master/studyTable/studyTable.png "study table") |

| `hingecabinet.xml` | `slidecabinet.xml` | `microwave.xml` |
| :---: | :---: | :---: |
| ![hingecabinet](https://raw.githubusercontent.com/MyoHub/furniture_sim/master/hingecabinet/hingecabinet.png "hinge cabinet") | ![slidecabinet](https://raw.githubusercontent.com/MyoHub/furniture_sim/master/slidecabinet/slidecabinet.png "slide cabinet") | ![microwave](https://raw.githubusercontent.com/MyoHub/furniture_sim/master/microwave/microwave.png "microwave") |

| `kitchen/kitchen1.xml`–`kitchen3.xml` | `oven.xml` | `counters.xml` |
| :---: | :---: | :---: |
| ![kitchen](https://raw.githubusercontent.com/MyoHub/furniture_sim/master/backwall/kitchen.png "kitchen") | ![oven](https://raw.githubusercontent.com/MyoHub/furniture_sim/master/oven/oven.png "oven") | ![counters](https://raw.githubusercontent.com/MyoHub/furniture_sim/master/counters/counters.png "counters") |

| `kettle.xml` (8 variants) | `kettle/kettle_body0.xml` | `bin.xml` |
| :---: | :---: | :---: |
| ![kettles](https://raw.githubusercontent.com/MyoHub/furniture_sim/master/kettle/kettles.png "kettles") | ![kettle](https://raw.githubusercontent.com/MyoHub/furniture_sim/master/kettle/kettle.png "kettle") | ![bin](https://raw.githubusercontent.com/MyoHub/furniture_sim/master/bin/bin.png "bin") |

Additional standalone models without preview images: `boxcabinet.xml`, `dishwasher.xml`, and `lightButtons.xml`.
