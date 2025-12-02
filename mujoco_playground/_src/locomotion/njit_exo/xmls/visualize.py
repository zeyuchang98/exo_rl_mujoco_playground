import mujoco
from mujoco import viewer

model = mujoco.MjModel.from_xml_path("/Users/zeyuchang/BioDynamic/mujoco_playground/mujoco_playground/_src/locomotion/njit_exo/xmls/scene_mjx_flat_terrain.xml")
data = mujoco.MjData(model)

home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

mujoco.mj_resetDataKeyframe(model, data, home_id)

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        v.sync()

print("qpos:", data.qpos)
print("ctrl:", data.ctrl)