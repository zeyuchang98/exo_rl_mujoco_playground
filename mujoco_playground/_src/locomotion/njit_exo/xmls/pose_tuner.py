import mujoco
import mujoco.viewer
import time

def get_custom_pose(xml_path="/Users/zeyuchang/BioDynamic/mujoco_playground/mujoco_playground/_src/locomotion/njit_exo/xmls/scene_mjx_flat_terrain.xml"):
    print("---------------------------------------------------------")
    print("INSTRUCTIONS:")
    print("1. The viewer will open paused.")
    print("2. Use the SLIDERS on the right (Control tab) to pose the legs.")
    print("3. Or hold CTRL + Right Click to drag limbs.")
    print("4. When happy, CLOSE the window to get your XML code.")
    print("---------------------------------------------------------")
    time.sleep(2)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Load your standing keyframe first as a base
    mujoco.mj_resetDataKeyframe(model, data, 0)
    key_id = 0
    model.qpos0[:] = model.key_qpos[key_id]
    data.qpos[:] = model.key_qpos[key_id]

    # Launch viewer
    mujoco.viewer.launch(model, data)

    # When viewer closes, this code runs:
    print("\n\n>>> COPY THIS INTO YOUR XML <keyframe> SECTION: <<<\n")
    
    # Format the qpos into a nice string
    qpos_str = " ".join([f"{x:.4f}" for x in data.qpos])
    
    # Split it for readability (Pos+Quat / Joints)
    # First 7 are freejoint, Rest are joints
    root = " ".join([f"{x:.4f}" for x in data.qpos[0:7]])
    joints = " ".join([f"{x:.4f}" for x in data.qpos[7:]])
    
    print(f'qpos="\n    {root}\n    {joints}"')
    print("\n---------------------------------------------------------")

if __name__ == "__main__":
    get_custom_pose()