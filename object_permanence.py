import bpy
import random
import math
import os

# -----------------------------------------------------------------------------
# User-configurable parameters
# -----------------------------------------------------------------------------
NUM_VIDEOS = 10000             # How many videos to generate
FPS = 24                    # Frames per second
VIDEO_SECONDS = 15           # Duration of each video (seconds)
FRAMES_PER_VIDEO = FPS * VIDEO_SECONDS
RESOLUTION_X = 256
RESOLUTION_Y = 256

OUTPUT_FOLDER = "./output"  # Change to your desired path
RENDER_PREFIX = "video"

# Number of terrain cubes to scatter
NUM_TERRAIN_CUBES = 30

# Bounding box size for terrain
TERRAIN_BOUNDS = 40

# 1–3 falling cubes
MIN_FALLING_CUBES = 1
MAX_FALLING_CUBES = 3

# Camera animation chance and duration
CAMERA_ANIMATION_CHANCE = 0.005   # 1/100 frames chance
CAMERA_ANIMATION_FRAMES = 15
CAMERA_ANIMATION_ANGLE = math.radians(90.0)  # Rotate ±90 degrees from current angle

# Ensure output folder exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# -----------------------------------------------------------------------------
# Utility: Delete all objects in the current scene
# -----------------------------------------------------------------------------
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Also remove leftover data-blocks to avoid accumulation
    for block in bpy.data.meshes:
        if not block.users:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if not block.users:
            bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if not block.users:
            bpy.data.images.remove(block)

# -----------------------------------------------------------------------------
# Setup camera rig, lighting, and render engine
# -----------------------------------------------------------------------------
def setup_scene():
    """
    Sets up the render engine (EEVEE Next), environment, 
    area lights, and creates a CameraRig (Empty at origin + Camera as child).
    """
    scene = bpy.context.scene

    # Use EEVEE Next
    scene.render.engine = 'BLENDER_EEVEE_NEXT'

    # EEVEE settings
    scene.eevee.use_raytracing = True
    scene.eevee.use_shadows = True
    scene.eevee.use_volumetric_shadows = True
    scene.eevee.use_fast_gi = True

    scene.eevee.shadow_ray_count = 4
    scene.eevee.shadow_step_count = 4

    scene.eevee.taa_render_samples = 16

    # Filmic view transform
    bpy.context.scene.view_settings.view_transform = 'Filmic'

    # Resolution and FPS
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.fps = FPS

    # Timeline range
    scene.frame_start = 1
    scene.frame_end = FRAMES_PER_VIDEO

    # Remove any existing camera(s)
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Remove any existing lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Create an Empty at the origin (CameraRig)
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    camera_rig = bpy.context.active_object
    camera_rig.name = "CameraRig"
    camera_rig.rotation_mode = 'XYZ'  # Ensure Euler for easy Z-rotation
    camera_rig.rotation_euler = (0.0, 0.0, 0.0)  # Start at 0 rotation

    # Create a camera at the desired location/rotation and parent to the rig
    bpy.ops.object.camera_add(
        location=(-35, -35, 30),
        rotation=(math.radians(60), 0, math.radians(-45))
    )
    camera = bpy.context.active_object
    camera.name = "Camera"
    camera.data.lens = 100
    # Parent to the rig, keeping transform
    camera.parent = camera_rig
    camera.matrix_parent_inverse = camera_rig.matrix_world.inverted()

    # Set active camera
    scene.camera = camera

    # Add two large area lights
    bpy.ops.object.light_add(type='AREA', 
                             location=(-45, 45, 60), 
                             rotation=(math.radians(-45), 0, math.radians(45)))
    area_light1 = bpy.context.active_object
    area_light1.data.energy = 100000.0
    area_light1.data.size = 5.0

    bpy.ops.object.light_add(type='AREA', 
                             location=(45, -45, 60), 
                             rotation=(math.radians(-45), 0, math.radians(-135)))
    area_light2 = bpy.context.active_object
    area_light2.data.energy = 100000.0
    area_light2.data.size = 5.0

    # World background to black
    world = bpy.data.worlds['World']
    world.use_nodes = True
    if "Background" in world.node_tree.nodes:
        world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

# -----------------------------------------------------------------------------
# Create random white "terrain" made of boxes
# -----------------------------------------------------------------------------
def create_random_terrain():
    """
    Creates randomly sized and positioned boxes and sets them as passive rigid bodies.
    Also creates a large base plane cube.
    Returns a list of all created terrain objects for intersection checks.
    """
    # Material for the terrain (white)
    mat_terrain = bpy.data.materials.new("TerrainMaterial")
    mat_terrain.diffuse_color = (1, 1, 1, 1)

    terrain_objects = []

    # Create base terrain cube
    bpy.ops.mesh.primitive_cube_add()
    base_cube = bpy.context.active_object
    base_cube.scale = (TERRAIN_BOUNDS, TERRAIN_BOUNDS, 0.5)
    base_cube.location = (0, 0, 0)
    if not base_cube.data.materials:
        base_cube.data.materials.append(mat_terrain)
    else:
        base_cube.data.materials[0] = mat_terrain

    bpy.ops.rigidbody.object_add()
    base_cube.rigid_body.type = 'PASSIVE'
    base_cube.rigid_body.collision_shape = 'BOX'
    terrain_objects.append(base_cube)

    # Scatter random boxes
    for _ in range(random.randint(NUM_TERRAIN_CUBES // 2, NUM_TERRAIN_CUBES)):
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object

        # Random dimensions
        width = random.uniform(2, 16)
        depth = random.uniform(2, 16)
        height = random.uniform(2, 8)
        cube.scale = (width / 2.0, depth / 2.0, height / 2.0)

        # Random position
        x = random.uniform(-TERRAIN_BOUNDS, TERRAIN_BOUNDS)
        y = random.uniform(-TERRAIN_BOUNDS, TERRAIN_BOUNDS)
        cube.location = (x, y, height / 2.0)

        # Assign white material
        if not cube.data.materials:
            cube.data.materials.append(mat_terrain)
        else:
            cube.data.materials[0] = mat_terrain

        # Passive rigid body
        bpy.ops.rigidbody.object_add()
        cube.rigid_body.type = 'PASSIVE'
        cube.rigid_body.collision_shape = 'BOX'

        terrain_objects.append(cube)

    return terrain_objects
# -----------------------------------------------------------------------------
# Create falling colored cubes
# -----------------------------------------------------------------------------
def create_falling_cubes(terrain_objects):
    """
    Spawns 1–3 cubes near the top with unique (R,G,B) colors and random rotation,
    ensuring they don't overlap with any terrain objects.
    """
    num_cubes = random.randint(MIN_FALLING_CUBES, MAX_FALLING_CUBES)

    colors = {
        "RED":   (1, 0, 0, 1),
        "GREEN": (0, 1, 0, 1),
        "BLUE":  (0, 0, 1, 1)
    }

    available_colors = list(colors.keys())
    used_positions = []
    cube_size = 2
    min_distance = cube_size * 1.5

    # Precompute bounding boxes for existing terrain
    terrain_bbs = [get_bounding_box(obj) for obj in terrain_objects]

    falling_cubes = []  # Track created falling cubes

    for i in range(num_cubes):
        # Pick a unique color
        color_name = random.choice(available_colors)
        available_colors.remove(color_name)
        color_val = colors[color_name]

        mat = bpy.data.materials.new(f"CubeColor_{color_name}")
        mat.diffuse_color = color_val
        mat.roughness = 0.5

        # Find a valid (x,y,z) that does NOT intersect with terrain or other falling cubes
        for attempt in range(100):  # up to 100 tries
            x = random.uniform(-6, 6)
            y = random.uniform(-6, 6)
            z = random.uniform(6, 10)
            
            # The falling cube has scale=(1,1,1) => half-extents = 1
            test_bb = (x - 1, x + 1, y - 1, y + 1, z - 1, z + 1)

            # Check against terrain
            intersects = False
            for bb in terrain_bbs:
                if boxes_intersect(test_bb, bb):
                    intersects = True
                    break

            # Check against previously placed falling cubes
            if not intersects:
                for cube in falling_cubes:
                    cube_bb = get_bounding_box(cube)
                    if boxes_intersect(test_bb, cube_bb):
                        intersects = True
                        break

            if not intersects:
                # Found a valid position, break out of attempts
                break
        else:
            # If we reach here (the for-else) with no 'break',
            # we failed to find a valid spawn in 100 tries.
            print("Warning: Could not find a valid non-intersecting location.")
            continue

        # Actually create the cube
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        cube.scale = (1, 1, 1)
        # Random rotation
        cube.rotation_euler = (
            random.random() * 2 * math.pi,
            random.random() * 2 * math.pi,
            random.random() * 2 * math.pi
        )
        cube.location = (x, y, z)

        # Assign color material
        if not cube.data.materials:
            cube.data.materials.append(mat)
        else:
            cube.data.materials[0] = mat

        # Dynamic rigid body
        bpy.ops.rigidbody.object_add()
        cube.rigid_body.type = 'ACTIVE'
        cube.rigid_body.collision_shape = 'BOX'
        cube.rigid_body.mass = 1.0

        falling_cubes.append(cube)

# -----------------------------------------------------------------------------
# Animate the camera rig's Z-rotation from current angle to current angle ± 90°
# over 15 frames, and stay there.
# -----------------------------------------------------------------------------
def animate_camera_rotation(rig, start_frame, direction):
    """
    Creates two keyframes on the rig's rotation_euler[2] (Z axis):
      - At start_frame: rotation = current angle
      - At start_frame + CAMERA_ANIMATION_FRAMES: rotation = current angle ± 90°

    We do not return to 0; the camera remains at ±90 from current angle forward.
    """
    scene = bpy.context.scene
    old_frame = scene.frame_current

    # Current angle on the rig's Z
    current_angle = rig.rotation_euler[2]

    # Choose direction
    angle_delta = CAMERA_ANIMATION_ANGLE  # 90 deg in radians
    if direction == 'RIGHT':
        angle_delta = -angle_delta

    start_angle = current_angle
    end_angle = current_angle + angle_delta

    # Keyframe 1: at start_frame
    scene.frame_set(start_frame)
    rig.rotation_euler[2] = start_angle
    rig.keyframe_insert(data_path="rotation_euler", index=2)

    # Keyframe 2: at start_frame + CAMERA_ANIMATION_FRAMES
    end_frame = start_frame + CAMERA_ANIMATION_FRAMES
    scene.frame_set(end_frame)
    rig.rotation_euler[2] = end_angle
    rig.keyframe_insert(data_path="rotation_euler", index=2)

    action = rig.animation_data.action
    fcurves = [fc for fc in action.fcurves if fc.data_path == "rotation_euler"]
    time = [start_frame, end_frame] # frame
    for fcurve in fcurves:
        # iterate thru keyframe Points and change easing of those at frame = time
        for kfp in fcurve.keyframe_points:
            # ('AUTO', 'EASE_IN', 'EASE_OUT', 'EASE_IN_OUT')
            if kfp.co.x in time:
                kfp.easing = 'EASE_IN_OUT' #  auto 

    # Restore old frame
    scene.frame_set(old_frame)

# -----------------------------------------------------------------------------
# Render animation frames (manually stepping frames for physics),
# with random camera swerves.
# -----------------------------------------------------------------------------
def render_animation(video_index):
    scene = bpy.context.scene

    # Ensure we have a Rigid Body World
    if not scene.rigidbody_world:
        bpy.ops.rigidbody.world_add()
    scene.rigidbody_world.point_cache.frame_end = scene.frame_end

    # Prepare output folder for this video
    output_path = os.path.join(OUTPUT_FOLDER, f"{RENDER_PREFIX}_{video_index}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Set image settings
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.quality = 90

    # Grab the camera rig
    camera_rig = bpy.data.objects.get("CameraRig", None)

    # Track whether the camera is currently in an active rotation
    camera_animation_end = 0

    # Prepare a log array: for each frame, store [0,0] or [1,0] or [0,1]
    log_data = [[0, 0] for _ in range(FRAMES_PER_VIDEO)]

    # Step through each frame
    for frame in range(scene.frame_start, scene.frame_end + 1):
        # Possibly trigger a new rotation if:
        #   1) we're past the current animation, AND
        #   2) there's enough frames left for a 15-frame animation.
        if (frame > camera_animation_end) and (frame + CAMERA_ANIMATION_FRAMES <= scene.frame_end):
            # 1/100 chance
            if random.random() < CAMERA_ANIMATION_CHANCE:
                direction = random.choice(['LEFT', 'RIGHT'])
                animate_camera_rotation(camera_rig, frame, direction)
                camera_animation_end = frame + CAMERA_ANIMATION_FRAMES
                # Log the start of this animation
                if direction == 'LEFT':
                    log_data[frame - 1] = [1, 0]
                else:
                    log_data[frame - 1] = [0, 1]

        # Advance physics
        scene.frame_set(frame)

        # Render this frame
        scene.render.filepath = os.path.join(output_path, f"frame_{frame:04d}.jpg")
        bpy.ops.render.render(write_still=True)

    # Write the log file: "video_{video_index-1:03d}.txt"
    # so video 1 => "video_000.txt", video 2 => "video_001.txt", etc.
    log_filename = f"video_{(video_index - 1):03d}.txt"
    log_path = os.path.join(output_path, log_filename)
    with open(log_path, 'w') as f:
        for row in log_data:
            f.write(f"[{row[0]},{row[1]}]\n")

# -----------------------------------------------------------------------------
# Main generation loop
# -----------------------------------------------------------------------------
def main():
    for i in range(1, NUM_VIDEOS + 1):
        print(f"--- Generating video {i}/{NUM_VIDEOS} ---")
        clear_scene()

        setup_scene()
        
        # Get the terrain objects so we can avoid them
        terrain_objects = create_random_terrain()
        
        # Pass them into create_falling_cubes
        create_falling_cubes(terrain_objects)

        render_animation(i)

        print(f"--- Finished video {i}/{NUM_VIDEOS} ---")

def get_bounding_box(obj):
    """
    Returns the (xmin, xmax, ymin, ymax, zmin, zmax) bounding box 
    for the given object's location/scale, assuming the object 
    is a uniform box at rotation_euler = (0,0,0).
    """
    x, y, z = obj.location
    sx, sy, sz = obj.scale  # half-extents because Blender box scale is half-size
    return (x - sx, x + sx,
            y - sy, y + sy,
            z - sz, z + sz)

def boxes_intersect(bb1, bb2):
    """
    Given two bounding boxes [xmin, xmax, ymin, ymax, zmin, zmax],
    returns True if they overlap in 3D space.
    """
    return not (
        bb1[1] < bb2[0] or  # A.maxX < B.minX
        bb1[0] > bb2[1] or  # A.minX > B.maxX
        bb1[3] < bb2[2] or  # A.maxY < B.minY
        bb1[2] > bb2[3] or  # A.minY > B.maxY
        bb1[5] < bb2[4] or  # A.maxZ < B.minZ
        bb1[4] > bb2[5]     # A.minZ > B.maxZ
    )

if __name__ == "__main__":
    main()
