import bpy
import random
import math
import os
import mathutils

# -----------------------------------------------------------------------------
# User-configurable parameters
# -----------------------------------------------------------------------------
NUM_VIDEOS = 10000             # How many videos to generate
FPS = 24                       # Frames per second
VIDEO_SECONDS = 15             # Duration of each video (seconds)
FRAMES_PER_VIDEO = FPS * VIDEO_SECONDS
RESOLUTION_X = 256
RESOLUTION_Y = 256

OUTPUT_FOLDER = "./output"  
RENDER_PREFIX = "video"

# Number of terrain cubes to scatter
NUM_TERRAIN_CUBES = 30

# Bounding box size for terrain
TERRAIN_BOUNDS = 40

# Falling cubes
MIN_FALLING_CUBES = 1
MAX_FALLING_CUBES = 3

# Camera animation chance and duration
CAMERA_ANIMATION_CHANCE = 0.005  
CAMERA_ANIMATION_FRAMES = 15
CAMERA_ANIMATION_ANGLE = math.radians(90.0)

# Click parameters
CLICKS_PER_VIDEO = 10000
MISS_LOG_PERCENT = 0.1  # 10% of misses get logged
EXPLOSIVE_FORCE = 50.0   # Increased from 5.0 to 50.0 for more visible effect

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
    camera_rig.rotation_mode = 'XYZ'
    camera_rig.rotation_euler = (0.0, 0.0, 0.0)

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
# Bounding box (for overlap checks)
# -----------------------------------------------------------------------------
def get_bounding_box(obj):
    """
    Returns the (xmin, xmax, ymin, ymax, zmin, zmax) bounding box 
    for the given object's location/scale, assuming no rotation or a simple box.
    """
    x, y, z = obj.location
    sx, sy, sz = obj.scale
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

# -----------------------------------------------------------------------------
# Create falling colored cubes
# -----------------------------------------------------------------------------
def create_falling_cubes(terrain_objects):
    """
    Spawns 1–3 cubes with random positions near the top, 
    random rotation, colored (red, green, blue).
    Returns a list of the falling cubes.
    """
    num_cubes = random.randint(MIN_FALLING_CUBES, MAX_FALLING_CUBES)

    colors = {
        "RED":   (1, 0, 0, 1),
        "GREEN": (0, 1, 0, 1),
        "BLUE":  (0, 0, 1, 1)
    }

    available_colors = list(colors.keys())
    falling_cubes = []

    # Precompute bounding boxes for existing terrain
    terrain_bbs = [get_bounding_box(obj) for obj in terrain_objects]

    for i in range(num_cubes):
        color_name = random.choice(available_colors)
        available_colors.remove(color_name)
        color_val = colors[color_name]

        mat = bpy.data.materials.new(f"CubeColor_{color_name}")
        mat.diffuse_color = color_val
        mat.roughness = 0.5

        # Attempt to find non-overlapping spawn
        for attempt in range(100):
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
                break
        else:
            print("Warning: Could not find a valid non-intersecting location.")
            continue

        # Create the cube
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        cube.scale = (1, 1, 1)
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

    return falling_cubes

# -----------------------------------------------------------------------------
# Animate the camera rig's Z-rotation
# -----------------------------------------------------------------------------
def animate_camera_rotation(rig, start_frame, direction):
    scene = bpy.context.scene
    old_frame = scene.frame_current

    current_angle = rig.rotation_euler[2]
    angle_delta = CAMERA_ANIMATION_ANGLE if direction == 'LEFT' else -CAMERA_ANIMATION_ANGLE

    start_angle = current_angle
    end_angle = current_angle + angle_delta

    # Keyframe 1
    scene.frame_set(start_frame)
    rig.rotation_euler[2] = start_angle
    rig.keyframe_insert(data_path="rotation_euler", index=2)

    # Keyframe 2
    end_frame = start_frame + CAMERA_ANIMATION_FRAMES
    scene.frame_set(end_frame)
    rig.rotation_euler[2] = end_angle
    rig.keyframe_insert(data_path="rotation_euler", index=2)

    # Make them ease in/out
    action = rig.animation_data.action
    fcurves = [fc for fc in action.fcurves if fc.data_path == "rotation_euler"]
    key_times = [start_frame, end_frame]
    for fcurve in fcurves:
        for kfp in fcurve.keyframe_points:
            if kfp.co.x in key_times:
                kfp.easing = 'EASE_IN_OUT'

    scene.frame_set(old_frame)

# -----------------------------------------------------------------------------
# Small helper: Construct a ray from the camera through pixel (px, py)
# and return (origin, direction).  This is a simplistic perspective approach.
# -----------------------------------------------------------------------------
def build_camera_ray(scene, camera_obj, px, py):
    """
    Convert screen coords (px,py) in [0..RESOLUTION_X, 0..RESOLUTION_Y]
    into a world-space ray (origin, direction). 
    """
    # Normalized device coords in [-1..1]
    ndc_x = (px / RESOLUTION_X) * 2.0 - 1.0
    ndc_y = (py / RESOLUTION_Y) * 2.0 - 1.0
    # In Blender camera space, +Y is "up," so we invert ndc_y
    # because typically +Y in NDC is top of image.
    ndc_y = -ndc_y

    # Camera matrix
    cam_mat = camera_obj.matrix_world
    # Focal length (mm)
    focal_mm = camera_obj.data.lens
    sensor_width = camera_obj.data.sensor_width  # default ~36mm
    # The ratio of ndc_x to sensor coords.  This is a simplified approach.
    # A more accurate approach would also consider camera_obj.data.sensor_fit.
    view_factor = (focal_mm / sensor_width)
    
    # In camera local space:
    # Z is forward, X is to the right, Y is up.
    # Let's define a local point at Z=-focal (in front of the camera),
    # with X/Y offset from ndc_x/ndc_y.
    # This scaling is somewhat arbitrary for a rough perspective ray.
    local_z = -1.0  # 1 Blender unit in front, negative because camera looks -Z
    local_x = ndc_x / view_factor
    local_y = ndc_y / view_factor

    local_point = mathutils.Vector((local_x, local_y, local_z))
    origin_local = mathutils.Vector((0.0, 0.0, 0.0))

    # Transform these to world space
    world_point = cam_mat @ local_point
    world_origin = cam_mat @ origin_local

    direction = (world_point - world_origin).normalized()
    return (world_origin, direction)

# -----------------------------------------------------------------------------
# Attempt a click at pixel (px,py).  If it hits one of the falling cubes,
# apply a small explosive impulse.  Return True if hit a falling cube, else False.
# -----------------------------------------------------------------------------
def try_click(scene, camera_obj, px, py, falling_cubes):
    # Build the ray
    origin, direction = build_camera_ray(scene, camera_obj, px, py)

    print(f"Origin: {origin}, Direction: {direction}")

    # Do the raycast
    result = scene.ray_cast(
        bpy.context.view_layer, 
        origin, 
        direction, 
        distance=9999.0
    )

    hit, hit_location, hit_normal, face_index, hit_object, matrix = result
    if not hit:
        return False
    
    print(f"Hit: {hit}, Location: {hit_location}, Normal: {hit_normal}, Face Index: {face_index}, Object: {hit_object}, Matrix: {matrix}")

    # Check if the object is one of the falling cubes
    if hit_object in falling_cubes:
        print("Hit a falling cube!")
        # Apply explosive force: from hit_location to object center
        obj_center = hit_object.location
        to_center = (obj_center - hit_location).normalized()
        
        # Add some upward bias to the force
        to_center.z += 0.5  # Add upward component
        to_center.normalize()  # Renormalize after adding upward bias
        
        # Add random variation to make it more dynamic
        random_variation = mathutils.Vector((
            random.uniform(-0.2, 0.2),
            random.uniform(-0.2, 0.2),
            random.uniform(0, 0.2)
        ))
        
        # Apply force with variation
        force_dir = (to_center + random_variation).normalized()
        hit_object.rigid_body.linear_velocity += force_dir * EXPLOSIVE_FORCE
        
        # Also add some angular velocity for spin
        hit_object.rigid_body.angular_velocity = mathutils.Vector((
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(-5, 5)
        ))
        
        return True

    return False


# -----------------------------------------------------------------------------
# Render animation frames (manually stepping frames),
# with random camera swerves and random clicks.
# -----------------------------------------------------------------------------
def render_animation(video_index, falling_cubes):
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

    # Grab the camera rig and camera
    camera_rig = bpy.data.objects.get("CameraRig", None)
    camera_obj = scene.camera

    # Prepare a log array with 4 columns per frame:
    # [rotLeft, rotRight, clickX, clickY]
    log_data = [[0, 0, 0, 0] for _ in range(FRAMES_PER_VIDEO)]

    # Decide in advance which frames will get a click (exactly 100).
    # We allow duplicates if random.sample < frames. If you want guaranteed unique frames:
    # frames_for_click = random.sample(range(1, FRAMES_PER_VIDEO+1), k=CLICKS_PER_VIDEO)
    # If duplicates are okay:
    frames_for_click = [i for i in range(1, FRAMES_PER_VIDEO+1)]

    # Keep track of camera rotation end to avoid overlapping animations
    camera_animation_end = 0

    # Step through each frame
    for frame in range(scene.frame_start, scene.frame_end + 1):

        print(f"Frame: {frame}")
        # Possibly trigger a new rotation if:
        #   (a) we're past the current animation
        #   (b) there's enough frames left
        if (frame > camera_animation_end) and (frame + CAMERA_ANIMATION_FRAMES <= scene.frame_end):
            if random.random() < CAMERA_ANIMATION_CHANCE:
                direction = random.choice(['LEFT', 'RIGHT'])
                animate_camera_rotation(camera_rig, frame, direction)
                camera_animation_end = frame + CAMERA_ANIMATION_FRAMES
                # Mark rotation in the log
                if direction == 'LEFT':
                    log_data[frame - 1][0] = 1  # left
                else:
                    log_data[frame - 1][1] = 1  # right

        # Advance physics
        scene.frame_set(frame)

        # Check if this frame is in frames_for_click
        if frame in frames_for_click:

            print(f"Clicking at frame {frame}")
            # Do one click
            px = random.randint(0, 255)
            py = random.randint(0, 255)
            hit_cube = try_click(scene, camera_obj, px, py, falling_cubes)
            if hit_cube:
                # Log it: put px in [2], py in [3]
                log_data[frame - 1][2] = px
                log_data[frame - 1][3] = py
            else:
                # We missed. Only 10% chance we explicitly show a (0,0) click
                if random.random() < MISS_LOG_PERCENT:
                    log_data[frame - 1][2] = px
                    log_data[frame - 1][3] = py
                    # Technically it is the same as "no click," 
                    # but we set it to [2]=0,[3]=0 anyway
                    # (they are already 0, so effectively no change).
                    pass

        # Render
        scene.render.filepath = os.path.join(output_path, f"frame_{frame:04d}.jpg")
        bpy.ops.render.render(write_still=True)

    # Write the log file
    # E.g. "video_000.txt" for video_index=1
    log_filename = f"video_{(video_index - 1):03d}.txt"
    log_path = os.path.join(output_path, log_filename)
    with open(log_path, 'w') as f:
        for row in log_data:
            # row is [rotLeft, rotRight, clickX, clickY]
            f.write(f"[{row[0]},{row[1]},{row[2]},{row[3]}]\n")

# -----------------------------------------------------------------------------
# Main generation loop
# -----------------------------------------------------------------------------
def main():
    for i in range(1, NUM_VIDEOS + 1):
        print(f"--- Generating video {i}/{NUM_VIDEOS} ---")
        clear_scene()

        setup_scene()
        
        # Create random terrain
        terrain_objects = create_random_terrain()
        
        # Create falling cubes
        falling_cubes = create_falling_cubes(terrain_objects)

        # Render & log
        render_animation(i, falling_cubes)

        print(f"--- Finished video {i}/{NUM_VIDEOS} ---")


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
