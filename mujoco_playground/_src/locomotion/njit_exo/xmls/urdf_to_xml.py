import mujoco
input_urdf = "./KF_NJIT_rehab_exo_novisual_boxcollision.urdf"
output_xml = "njit_exo.xml"

print(f"Loading {input_urdf}...")
model = mujoco.MjModel.from_xml_path(input_urdf)

print("Conversion successful. Saving MJCF...")
mujoco.mj_saveLastXML(output_xml, model)

print(f"Done! Saved as {output_xml}")
print("Update your 'base.py' or config to point to this new .xml file.")



from dm_control import mjcf
from dm_control.mjcf import exporter

urdf_path = "/Users/zeyuchang/BioDynamic/mujoco_playground/mujoco_playground/_src/locomotion/njit_exo/xmls/KF_NJIT_rehab_exo_novisual_boxcollision_fix.urdf"
mjcf_model = mjcf.from_urdf(urdf_path)

xml_string = mjcf_model.to_xml_string()
with open("robot_converted.xml", "w") as f:
    f.write(xml_string)

#--------------------------------------------

import mujoco
import mujoco.viewer

# MuJoCo automatically parses URDFs if the file extension is .urdf 
# or if the root tag is <robot>
model = mujoco.MjModel.from_xml_path("/Users/zeyuchang/BioDynamic/mujoco_playground/mujoco_playground/_src/locomotion/njit_exo/xmls/scene_mjx_flat_terrain.xml")
data = mujoco.MjData(model)

name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 0)
print(name)
# Launch the viewer to see it
mujoco.viewer.launch(model, data)


import mujoco
# Read your URDF text
oringinal_urdf_path = '/Users/zeyuchang/BioDynamic/mujoco_playground/mujoco_playground/_src/locomotion/njit_exo/xmls/KF_NJIT_rehab_exo_novisual_boxcollision.urdf'
with open(oringinal_urdf_path, "r") as f:
    urdf_text = f.read()
# Inject compiler options WITHOUT modifying the actual file
wrapped = (
    '<mujoco><compiler fusestatic="false"/></mujoco>\n' + urdf_text
)
# Load from memory buffer
model = mujoco.MjModel.from_xml_string(wrapped)
# Save expanded MJCF
mujoco.mj_saveLastXML("./exoskeleton_no_fusestatic.xml", model)

m = mujoco.MjModel.from_xml_path("/Users/zeyuchang/BioDynamic/mujoco_playground/mujoco_playground/_src/locomotion/njit_exo/xmls/KF_NJIT_rehab_exo_novisual_boxcollision.urdf")
d = mujoco.MjData(m)
with mujoco.viewer.launch_passive(m, d) as viewer:
    # Disable the rendering of static geoms
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = False

    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()

#--------------------------------------------

import mujoco

spec = mujoco.MjSpec.from_file("/Users/zeyuchang/BioDynamic/mujoco_playground/mujoco_playground/_src/locomotion/njit_exo/xmls/KF_NJIT_rehab_exo_novisual_boxcollision.urdf")
spec.compiler.discardvisual = False 
xml_content = spec.to_xml()
with open("/Users/zeyuchang/BioDynamic/mujoco_playground/mujoco_playground/_src/locomotion/njit_exo/xmls/njit_exo.xml", "w") as f:
    f.write(xml_content)

# 4. If you also need the compiled binary model object for simulation:
model = spec.compile()


#--------------------------------------------
import mujoco

# 1. Load the model
xml_path = "/Users/zeyuchang/BioDynamic/mujoco_playground/mujoco_playground/_src/locomotion/njit_exo/xmls/scene_mjx_flat_terrain.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 2. Step the physics once to ensure everything settles (optional)
mujoco.mj_step(model, data)

# 3. Create a renderer
renderer = mujoco.Renderer(model)

# 4. Update the scene and render
renderer.update_scene(data)
pixels = renderer.render()

# 5. Save the image
# If you don't have the 'media' library, use Pillow:
from PIL import Image
img = Image.fromarray(pixels)
img.save("output_view.png")

print("Image saved to output_view.png")

#--------------------------------------------
import mujoco
import mujoco.viewer
from PIL import Image

model = mujoco.MjModel.from_xml_path('/Users/zeyuchang/BioDynamic/mujoco_playground/mujoco_playground/_src/locomotion/njit_exo/xmls/scene_mjx_flat_terrain.xml')
data = mujoco.MjData(model)

# 2. Snap the robot to the "home" keyframe (ID 0)
# This applies the standing position we defined in the XML
mujoco.mj_resetDataKeyframe(model, data, 0)
key_id = 0
model.qpos0[:] = model.key_qpos[key_id]
# 3. Set the current state to match (so it starts correctly)
data.qpos[:] = model.key_qpos[key_id]
mujoco.mj_forward(model, data) 

# 3. Print the height to confirm it is correct
print(f"Robot Start Height: {data.qpos[2]} meters")

# 4. Setup a specific camera view so the robot is in frame
camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(camera)
camera.distance = 3.0       # 3 meters away
camera.azimuth = 90         # Side view
camera.elevation = -10      # Slight downward angle
camera.lookat = [0, 0, 1.0] # Look at the robot's center (approx 1m height)

# 5. Render using the custom camera
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera=camera)
pixels = renderer.render()

# 6. Save the picture
image = Image.fromarray(pixels)
image.save("initial_position.png")

#------------
# --------------------------------
import mujoco
import mujoco.viewer

# 1. Load the model
model = mujoco.MjModel.from_xml_path('/Users/zeyuchang/BioDynamic/mujoco_playground/mujoco_playground/_src/locomotion/njit_exo/xmls/scene_mjx_flat_terrain.xml')
# model = mujoco.MjModel.from_xml_path('/Users/zeyuchang/BioDynamic/mujoco_playground/mujoco_playground/_src/locomotion/njit_exo/xmls/njit_exo.xml')
# model = mujoco.MjModel.from_xml_path('/Users/zeyuchang/BioDynamic/mujoco_playground/mujoco_playground/_src/locomotion/njit_exo/xmls/KF_NJIT_rehab_exo_novisual_boxcollision.urdf')

data = mujoco.MjData(model)

# 2. VITAL STEP: Reset the state to your "home" keyframe (ID 0)
# If you skip this, it will spawn at (0,0,0) instead of your QPOS.
mujoco.mj_resetDataKeyframe(model, data, 0)

# key_id = 0
# model.qpos0[:] = model.key_qpos[key_id]

# # 3. Set the current state to match (so it starts correctly)
# data.qpos[:] = model.key_qpos[key_id]
# 3. Launch the viewer
# The simulation will be paused at the start so you can check the position.
mujoco.viewer.launch(model, data)


#--------------------------------------------
import time
import mujoco
import mujoco.viewer

# 1. Load your model
model = mujoco.MjModel.from_xml_path("/Users/zeyuchang/BioDynamic/mujoco_playground/mujoco_playground/_src/locomotion/njit_exo/xmls/scene_mjx_flat_terrain.xml")
data = mujoco.MjData(model)

# 2. Manage the pause state
paused = True

def key_callback(keycode):
    """
    Toggle the pause state when the Space bar is pressed.
    """
    if chr(keycode) == ' ':
        global paused
        paused = not paused

# 3. Launch the passive viewer
# This opens the window but does NOT automatically run the physics loop.
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    
    # Close the viewer automatically if the user closes the window
    while viewer.is_running():
        step_start = time.time()

        # ONLY advance the physics if not paused
        if not paused:
            mujoco.mj_step(model, data)

        # Sync the viewer with the current physics state
        # This is necessary to render the frame even if paused
        viewer.sync()

        # Optional: Sleep to slow down to roughly real-time
        # (remove this if you want maximum speed)
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)