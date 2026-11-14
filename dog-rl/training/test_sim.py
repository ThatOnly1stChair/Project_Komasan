import mujoco    
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        start = time.time()

        data.ctrl = [
            0.00, 0.33, 0.883,
            0.00, 0.33, 0.883,
            0.00, 0.33, 0.883,
            0.00, 0.33, 0.883
        ]
        mujoco.mj_step(model, data)

        viewer.sync()

        elapsed = time.time() - start
        remaining = model.opt.timestep - elapsed

        if remaining > 0:
            time.sleep(remaining)