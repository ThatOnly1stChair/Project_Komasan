import mujoco
import mujoco.viewer
from pathlib import Path

# 1. Load your robot XML model
MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "scene.xml"
)
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# 2. Launch the interactive GUI viewer
mujoco.viewer.launch(model, data)
