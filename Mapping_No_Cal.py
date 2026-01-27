import cv2
import numpy as np
import os
print(os.curdir)
# ---------------------------------------------------------
# THE ALGORITHM: Phase 1 (Setup & Anchor)
# ---------------------------------------------------------

# State Variables
canvas = None         # The map we are building
canvas_size = (10000, 10000)
source_folder = r"C:\Users\Pc\Desktop\Yolo Kodlari\Otonom gorev\Test\Matris_Mapping\Proccessing"
print(f"--- Processing First Input: {source_folder} ---")
# 1. LOAD ALL IMAGES (Dynamic List)
# We need a list of all images to create the "Chain"

files = sorted([f for f in os.listdir(source_folder) if f.endswith((".jpg", ".png", ".jpeg", ".JPG"))])
print(files)
print(type(files[0]))
images = []
for f in files:
    img_path = os.path.join(source_folder, f)
    img = cv2.imread(img_path)
    if img is not None:
        images.append(img)
        print(f"Loaded: {f}")

if len(images) < 2:
    print("Error: Need at least 2 images.")
    exit()

# Set Image 1 as the Anchor
img1 = images[0]
img1_h, img1_w, _ = img1.shape



# 2. CREATE CANVAS
canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
print(f"Canvas Created: {canvas_size}")

# 3. CALCULATE ANCHOR POSITION (Your Phase 1 Logic)
center_y, center_x = canvas_size[0] // 2, canvas_size[1] // 2
start_y = center_y - (img1_h // 2)
start_x = center_x - (img1_w // 2)
end_y = start_y + img1_h
end_x = start_x + img1_w

if start_x < 0 or start_y < 0:
    print("CRITICAL ERROR: The image is still bigger than the canvas!")
    exit()

# We don't "burn" img1 into the canvas yet. We will render ALL images in Phase 3.
# This ensures layers are handled consistently.


# ---------------------------------------------------------
# THE ALGORITHM: Phase 2 (Batch Matrix Calculation)
# ---------------------------------------------------------
print("\n--- Phase 2: Calculating Global Matrices ---")

mapped_images = [(images[0], np.eye(3))]
current_global_H = np.eye(3) 

# Use BFMatcher (Better for this than FLANN)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
sift = cv2.SIFT_create(nfeatures=20000)

for i in range(len(images) - 1):
    img_prev = images[i]   # Target
    img_curr = images[i+1] # Source
    
    print(f"Matching Image {i+2} -> Image {i+1}...")

    # A. Detect Features
    kp_prev, des_prev = sift.detectAndCompute(img_prev, None)
    kp_curr, des_curr = sift.detectAndCompute(img_curr, None)
    print(f"  -> Features detected: Prev={len(kp_prev)}, Curr={len(kp_curr)}")

    print("des_prev", des_prev)
    # print("des_curr", des_curr)
    
    # --- CRITICAL FIX START ---
    # Check if features actually exist before trying to match
    if des_prev is None or des_curr is None:
        print(f"  -> ERROR: No features found in one of the images. Skipping stitch.")
        continue # Skip to the next pair
        
    if len(des_prev) < 2 or len(des_curr) < 2:
        print(f"  -> ERROR: Not enough features ({len(des_curr)} found). Needs 2+. Skipping.")
        continue
    # --- CRITICAL FIX END ---

    # B. Match Features (BFMatcher)
    try:
        matches = bf.knnMatch(des_curr, des_prev, k=2)
    except Exception as e:
        print(f"  -> Matcher Failed: {e}")
        continue

    # Filter matches
    good = []
    for m, n in matches:
        if m.distance < .7 * n.distance:
            good.append(m)
            
    if len(good) > 10: # Need at least 10 good points
        # C. Calculate Homography
        src_pts = np.float32([kp_curr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_prev[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        H_local, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        
        # D. Chain the Matrix
        if H_local is not None:
            # Convert 2x3 Affine to 3x3 Homography format
            row = np.array([[0, 0, 1]])
            H_local = np.vstack((H_local, row))
            
            current_global_H = np.dot(current_global_H, H_local)
            mapped_images.append((img_curr, current_global_H))
            print(f"  -> Success: Locked with {len(good)} matches.")
        else:
             print("  -> Warning: Matrix calculation failed.")
    else:
        print(f"  -> Warning: Not enough strong matches ({len(good)}/10).")
        # Optional: You might want to break here if the chain is broken
        # break


# ---------------------------------------------------------
# THE ALGORITHM: Phase 3 (Batch Render)
# ---------------------------------------------------------
print("\n--- Phase 3: Rendering to Canvas ---")

# 1. CREATE OFFSET MATRIX
# This moves everything from (0,0) to the center (start_x, start_y)
# We use the exact calculation from your Phase 1 code.
offset_matrix = np.array([
    [1, 0, start_x],
    [0, 1, start_y],
    [0, 0, 1]
])

# 2. RENDER LOOP
for idx, (img_data, H_global) in enumerate(mapped_images):
    print(f"Rendering Image {idx+1}/{len(mapped_images)}...")
    
    # Combine the Global Movement with the Center Offset
    final_render_matrix = np.dot(offset_matrix, H_global)
    
    # Warp the image directly onto the giant canvas
    warped_img = cv2.warpPerspective(img_data, final_render_matrix, (canvas_size[1], canvas_size[0]))
    
    # Masking: Only copy non-black pixels
    # (This prevents the black borders of warped images from covering previous ones)
    valid_mask = np.sum(warped_img, axis=2) > 0
    canvas[valid_mask] = warped_img[valid_mask]

# 3. CROPPING BLACK BORDERS
print("\n--- Phase 4: Cropping Black Borders ---")
gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
coords = cv2.findNonZero(gray_canvas)
if coords is not None:
    x, y, w, h = cv2.boundingRect(coords)
    print(f"Cropping to: x={x}, y={y}, w={w}, h={h}")
    canvas = canvas[y:y+h, x:x+w]
else:
    print("Warning: Canvas is empty, skipping crop.")

# 4. SAVE RESULT
output_file = "Phase3.jpg"
cv2.imwrite(output_file, canvas)
print(f"\nSuccess! Map saved as '{output_file}'")