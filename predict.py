import cv2
import numpy as np
from scipy.optimize import differential_evolution

# -----------------------
# Camera Setup
# -----------------------
width = 256
height = 256
focal_length_mm = 80       # Blender focal length in mm
sensor_width_mm = 36       # Blender sensor width (36mm for full-frame)

# Convert focal length from mm to pixels
focal_length_pixels = (focal_length_mm * width) / sensor_width_mm

# Camera position and target (for look-at)
cam_pos = np.array([-30, 25, -30])
target_pos = np.array([0.0, 0.0, 0.0])
up_vector = np.array([0.0, 1.0, 0.0])

def create_lookat_matrix(eye, target, up):
    """Create a look-at rotation and translation matrix."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)
    
    R = np.array([
        [right[0], right[1], right[2]],
        [new_up[0], new_up[1], new_up[2]],
        [-forward[0], -forward[1], -forward[2]]
    ])
    t = -R @ eye
    return R, t.reshape(3,1)

# Camera intrinsic matrix
K = np.array([[focal_length_pixels, 0, width/2],
              [0, focal_length_pixels, height/2],
              [0, 0, 1]])

# Camera extrinsic matrix using the look-at method
R_cam, t_cam = create_lookat_matrix(cam_pos, target_pos, up_vector)
RT_cam = np.hstack((R_cam, t_cam))

# -----------------------
# Cube Definition
# -----------------------
# Define cube vertices (2x2x2 cube centered at origin) in homogeneous coordinates.
cube_vertices = np.array([
    [-1.0, -1.0, -1.0, 1],
    [ 1.0, -1.0, -1.0, 1],
    [ 1.0,  1.0, -1.0, 1],
    [-1.0,  1.0, -1.0, 1],
    [-1.0, -1.0,  1.0, 1],
    [ 1.0, -1.0,  1.0, 1],
    [ 1.0,  1.0,  1.0, 1],
    [-1.0,  1.0,  1.0, 1]
]).T

# Define cube edges (indices into the vertex list)
edges = [
    (0,1), (1,2), (2,3), (3,0),  # bottom face
    (4,5), (5,6), (6,7), (7,4),  # top face
    (0,4), (1,5), (2,6), (3,7)   # vertical edges
]

# -----------------------
# Load the Image
# -----------------------
# Try to load the image (cubes.jpg) and resize it.
img = cv2.imread('./cubes.jpg')
if img is None:
    print("Error: Could not load cubes.jpg. Using a blank image.")
    img = np.zeros((height, width, 3), dtype=np.uint8)
else:
    img = cv2.resize(img, (width, height))
    
# We also keep a copy for mask extraction.
img_for_mask = img.copy()

# -----------------------
# Color Abstraction Functions
# -----------------------
def get_color_mask(image, color):
    """
    Returns a binary mask for the given color.
    For red, two HSV ranges are combined.
    For green and blue, one range is used (you may need to adjust these ranges).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color.lower() == "red":
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 100])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif color.lower() == "green":
        lower = np.array([40, 100, 100])
        upper = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif color.lower() == "blue":
        lower = np.array([100, 100, 100])
        upper = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    return mask

def get_color_features(image, color):
    """
    Given an image and a color name, compute:
      - The color mask.
      - Canny edges computed on the masked image.
      - A distance transform (DT) computed from the inverted edges.
      - Corners detected from the masked image.
    Returns: mask, edges, dt, corners
    """
    mask = get_color_mask(image, color)
    if cv2.countNonZero(mask) == 0:
        return mask, None, None, None

    # Use the mask to extract the color region.
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    
    # Compute edges on the masked image and filter for straight lines
    edges = cv2.Canny(masked_img, 50, 120)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=15, minLineLength=10, maxLineGap=10)
    edges = np.zeros_like(edges)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(edges, (x1,y1), (x2,y2), 255, 2)
    
    # Compute the distance transform from the inverted edge image.
    dt = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)
    
    # Detect corners on the masked grayscale image.
    gray_masked = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_masked, maxCorners=3, qualityLevel=0.25, minDistance=5)
    if corners is not None:
        corners = corners.reshape(-1, 2)
        
        # Remove corners that are midpoints between other corners
        corners_to_keep = []
        for i, corner in enumerate(corners):
            is_midpoint = False
            
            # Check if this corner is a midpoint between any other pair of corners
            for j in range(len(corners)):
                for k in range(j+1, len(corners)):
                    if j != i and k != i:  # Don't compare corner with itself
                        midpoint = (corners[j] + corners[k]) / 2
                        dist = np.sqrt(np.sum((corner - midpoint) ** 2))
                        if dist < 1:  # Small threshold for considering it a midpoint
                            is_midpoint = True
                            break
                if is_midpoint:
                    break
                    
            if not is_midpoint:
                corners_to_keep.append(corner)
                
        corners = np.array(corners_to_keep) if corners_to_keep else np.empty((0, 2))
    else:
        corners = np.empty((0, 2))

    # Visualize edges and corners for debugging
    debug_img = masked_img.copy()
    
    # Draw edges in green
    if edges is not None:
        debug_img[edges > 0] = [0, 255, 0]
    
    # Draw corners as red circles
    if corners is not None:
        for corner in corners:
            x, y = corner.astype(int)
            cv2.circle(debug_img, (x, y), 3, (0, 0, 255), -1)
    
    cv2.imshow('Edges and Corners', debug_img)
    cv2.waitKey(100)
    
    return mask, edges, dt, corners

# -----------------------
# Cube Projection Function
# -----------------------
def project_cube(params):
    """
    Project the cube given 6 parameters: [x, y, z, rx, ry, rz].
    [rx, ry, rz] is a Rodrigues rotation vector.
    """
    x, y, z, rx, ry, rz = params
    rvec = np.array([[rx], [ry], [rz]], dtype=np.float64)
    R_cube, _ = cv2.Rodrigues(rvec)
    t_cube = np.array([[x], [y], [z]], dtype=np.float64)
    
    # Build a 4x4 homogeneous transformation for the cube.
    RT_cube_4x4 = np.eye(4)
    RT_cube_4x4[:3, :3] = R_cube
    RT_cube_4x4[:3, 3] = t_cube.flatten()
    
    # Build a 4x4 homogeneous camera transformation.
    RT_cam_4x4 = np.eye(4)
    RT_cam_4x4[:3, :3] = R_cam
    RT_cam_4x4[:3, 3] = t_cam.flatten()
    
    # Projection matrix (3x4)
    P = K @ RT_cam[:3, :]
    
    # Project the cube vertices.
    projected = P @ RT_cube_4x4 @ cube_vertices
    projected /= projected[2]
    return projected[:2].T

# -----------------------
# Objective Function (Color-Agnostic)
# -----------------------
def objective_function(params, mask, edge_dt, corners):
    """
    Compute an objective score for the cube projection based on:
      - Overlap (IoU) with the mask.
      - A penalty for the distance of edge midpoints from image edges.
      - A penalty for the distance of cube vertices from detected corners,
        with stronger emphasis on close matches.
      - A penalty for non-flat orientations relative to the floor.
    """
    projected_points = project_cube(params)
    render = np.zeros((height, width), dtype=np.uint8)
    for start, end in edges:
        pt1 = tuple(map(int, projected_points[start]))
        pt2 = tuple(map(int, projected_points[end]))
        cv2.line(render, pt1, pt2, 255, 2)
    
    # Overlap with the mask.
    overlap = cv2.bitwise_and(render, mask)
    
    cube_area = float(np.sum(render)) / 255.0
    target_area = float(np.sum(mask)) / 255.0
    if cube_area == 0:
        return 1000  # Large penalty for an invisible cube
    
    overlap_area = float(np.sum(overlap)) / 255.0
    iou = overlap_area / (cube_area + target_area - overlap_area + 1e-6)
    # iou = overlap_area / target_area
    
    # --- Edge and Corner Matching Penalties ---
    edge_weight = 0.5
    edge_penalty = 0.0
    edge_distances = []
    for start, end in edges:
        midpoint = (projected_points[start] + projected_points[end]) / 2.0
        midpoint_int = np.round(midpoint).astype(int)
        if 0 <= midpoint_int[0] < width and 0 <= midpoint_int[1] < height:
            dist = edge_dt[midpoint_int[1], midpoint_int[0]]
            edge_distances.append(dist)
    
    # Take only the best N edge matches
    edge_distances.sort()  # Sort distances from best (lowest) to worst
    n_best_edges = 3  # Number of best edges to consider
    edge_penalty = sum(edge_distances[:n_best_edges]) * edge_weight * 3  # Increased weight for best matches

    vertex_weight = 0.0
    vertex_distances = []
    for vertex in projected_points:
        if corners.shape[0] > 0:
            distances = np.linalg.norm(corners - vertex, axis=1)
            min_dist = distances.min()
            vertex_distances.append(min_dist)
    
    # Take only the best N vertex matches
    vertex_distances.sort()  # Sort distances from best (lowest) to worst
    n_best_vertices = 3  # Number of best vertices to consider
    vertex_penalty = sum(vertex_distances[:n_best_vertices]) * vertex_weight * 3  # Increased weight for best matches

    total_score = -20 * iou + edge_penalty + vertex_penalty
    return total_score

# -----------------------
# Color Mapping for Drawing (BGR Order)
# -----------------------
color_bgr = {
    "red":   (0, 0, 255),
    "green": (0, 255, 0),
    "blue":  (255, 0, 0)
}

# -----------------------
# New Functionality: Cube Pose Estimation and Comparison
# -----------------------
colors = ["green", "red", "blue"]

def estimate_cube_poses(image):
    """
    Estimate optimized cube poses for each color in the provided image.
    Returns a dictionary mapping color names to optimized pose parameter arrays.
    """
    poses = {}
    # Use a copy of the image for analysis
    img_for_analysis = image.copy()
    for color in colors:
        print("\nProcessing", color, "cube...")
        mask, edges_img, dt, corners = get_color_features(img_for_analysis, color)
        
        if mask is None or cv2.countNonZero(mask) == 0 or dt is None:
            print(f"No {color} mask found. Skipping {color} cube optimization.")
            continue
        
        # Find initial parameters for this color
        found_initial = False
        attempt = 0
        init_params = None
        while not found_initial and attempt < 10000:
            attempt += 1
            init_params = np.array([
                np.random.uniform(-10, 10),       # x
                np.random.uniform(0, 8),          # y
                np.random.uniform(-10, 10),       # z
                np.random.uniform(-np.pi, np.pi), # rx
                np.random.uniform(-np.pi, np.pi), # ry
                np.random.uniform(-np.pi, np.pi)  # rz
            ])
            obj_val = objective_function(init_params, mask, dt, corners)
            print(f"Attempt {attempt}: obj = {obj_val:.2f}")
            if obj_val < 30:
                found_initial = True
                print(f"Found initial parameters for {color} cube on attempt {attempt} with objective {obj_val:.2f}:")
                print(init_params)
        
        if not found_initial:
            print(f"Could not find an initial pose for the {color} cube with sufficient overlap. Skipping.")
            continue
        
        # Optimize the pose for this color
        bounds = [
            (-10, 10),    # x
            (0, 10),      # y
            (-10, 10),    # z
            (-np.pi, np.pi),  # rx
            (-np.pi, np.pi),  # ry
            (-np.pi, np.pi)   # rz
        ]
        
        result = differential_evolution(
            objective_function,
            bounds,
            args=(mask, dt, corners),
            strategy='best1bin',
            maxiter=3000,
            popsize=50,
            mutation=(0.5, 1.0),
            recombination=0.7,
            tol=1e-5,
            atol=1e-5,
            disp=False,
            workers=-1  # Use all available CPU cores
        )
        
        optimized_params = result.x
        best_loss = result.fun
        print(f"Optimized parameters for {color} cube:", optimized_params)
        print(f"Final loss for {color} cube: {best_loss:.2f}")
        
        poses[color] = optimized_params
        
    return poses

def compare_cube_poses(image1, image2):
    """
    Compare cube poses between two images.
    Estimates cube poses for each image and computes the difference for each color.
    Returns a dictionary mapping each color to a difference metric, which includes:
      - difference_vector: The element-wise difference between the pose parameters.
      - difference_norm: The Euclidean norm of the difference vector.
    If a cube pose is not found in one of the images, returns None for that color.
    """
    poses1 = estimate_cube_poses(image1)
    poses2 = estimate_cube_poses(image2)
    # Normalize differences based on parameter bounds
    bounds = np.array([
        20.0,     # x: [-10,10] range = 20
        10.0,     # y: [0,10] range = 10  
        20.0,     # z: [-10,10] range = 20
        2*np.pi,  # rx: [-pi,pi] range = 2pi
        2*np.pi,  # ry: [-pi,pi] range = 2pi
        2*np.pi   # rz: [-pi,pi] range = 2pi
    ])
    
    total_diff = 0
    found_cubes = 0
    
    for color in colors:
        if color in poses1 and color in poses2:
            # Get normalized difference for each parameter
            diff_vector = (poses1[color] - poses2[color]) / bounds
            diff_norm = np.linalg.norm(diff_vector)
            total_diff += diff_norm
            found_cubes += 1
            
    # Return average normalized difference scaled to 0-20 range
    # If no cubes found, return max score of 20
    if found_cubes == 0:
        return 60.0
    else:
        return min(60.0, (total_diff / found_cubes) * 60.0)

# -----------------------
# Process Each Color (Main Execution)
# -----------------------
if __name__ == "__main__":
    final_img = img.copy()

    for color in colors:
        print("\nProcessing", color, "cube...")
        
        # Get the color mask and features.
        mask, edges_img, dt, corners = get_color_features(img_for_mask, color)
        
        # If the mask is empty, skip this color.
        if mask is None or cv2.countNonZero(mask) == 0 or dt is None:
            print(f"No {color} mask found. Skipping {color} cube optimization.")
            continue
        
        # Find initial parameters for this color
        found_initial = False
        attempt = 0
        init_params = None
        while not found_initial and attempt < 10000:
            attempt += 1
            init_params = np.array([
                np.random.uniform(-10, 10),       # x
                np.random.uniform(0, 8),          # y
                np.random.uniform(-10, 10),       # z
                np.random.uniform(-np.pi, np.pi), # rx
                np.random.uniform(-np.pi, np.pi), # ry
                np.random.uniform(-np.pi, np.pi)  # rz
            ])
            obj_val = objective_function(init_params, mask, dt, corners)
            
            print(f"Attempt {attempt}: obj = {obj_val:.2f}")
            if obj_val < 30:
                found_initial = True
                print(f"Found initial parameters for {color} cube on attempt {attempt} with objective {obj_val:.2f}:")
                print(init_params)
        
        if not found_initial:
            print(f"Could not find an initial pose for the {color} cube with sufficient overlap. Skipping.")
            continue
        
        # Optimize the pose for this color
        bounds = [
            (-10, 10),    # x
            (0, 10),      # y
            (-10, 10),    # z
            (-np.pi, np.pi),  # rx
            (-np.pi, np.pi),  # ry
            (-np.pi, np.pi)   # rz
        ]
        
        result = differential_evolution(
            objective_function,
            bounds,
            args=(mask, dt, corners),
            strategy='best1bin',
            maxiter=3000,
            popsize=50,
            mutation=(0.5, 1.0),
            recombination=0.7,
            tol=1e-5,
            atol=1e-5,
            disp=True,
            workers=-1  # Use all available CPU cores
        )

        optimized_params = result.x
        best_loss = result.fun
        print(f"\nOptimized parameters for {color} cube:", optimized_params)
        print(f"Final loss for {color} cube: {best_loss:.2f}")
        
        # Draw the optimized cube on the final image
        optimized_projection = project_cube(optimized_params)
        for start, end in edges:
            pt1 = tuple(map(int, optimized_projection[start]))
            pt2 = tuple(map(int, optimized_projection[end]))
            cv2.line(final_img, pt1, pt2, color_bgr[color], 2)
    
    # Show final image with all cubes
    cv2.imshow("Final Cube Fits", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
