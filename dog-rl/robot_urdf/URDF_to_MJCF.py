import mujoco

#Convert the Onshape assembly to a URDF
# onshape-to-robot robot_urdf

#After you create the XML file and add it to the scene.XML, you can visualize it with:
# onshape-to-robot-mujoco robot_urdf
#It will launch scene.xml in MuJoCo

URDF_file = "robot_urdf/robot.urdf"

model = mujoco.MjModel.from_xml_path(URDF_file)

print(f"bodies: {model.nbody}")
print(f"joints: {model.njnt}")
print(f"DOF: {model.nv}")

if model.nv == 12:
    spec = mujoco.MjSpec.from_file(URDF_file)
    spec_model = spec.compile()
    spec.encode("robot_urdf/robot.xml", model=spec_model)
    print("Model converted to xml!")