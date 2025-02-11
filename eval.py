import os
import sys
import cv2
import numpy as np
import lpips
import torch
from predict import compare_cube_poses

loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

def calculate_mse(img1, img2):
    """Calculate Mean Squared Error between two images"""
    if img1.shape != img2.shape:
        raise ValueError("Images must have same dimensions")
    mse = np.mean((img1 - img2) ** 2)
    print(f"Raw MSE: {mse}")
    print(f"Normalized MSE: {(mse / 255.0 / 255.0) * 20.0}")
    # Normalize MSE to 0-20 range
    return min(20.0, (mse / 255.0 / 255.0) * 20.0)

def calculate_lpips(img1, img2):
    """Calculate LPIPS perceptual distance between two images"""
    # Convert BGR to RGB and normalize to [-1,1]
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1 = (img1 / 127.5) - 1
    img2 = (img2 / 127.5) - 1
    
    # Convert to torch tensors
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
    
    # Calculate LPIPS and normalize to 0-20
    with torch.no_grad():
        lpips_score = float(loss_fn_vgg(img1, img2))
    return min(20.0, lpips_score * 20.0)

def main():
    if len(sys.argv) != 3:
        print("Usage: python eval.py <folder1> <folder2>")
        sys.exit(1)
        
    folder1 = sys.argv[1]
    folder2 = sys.argv[2]
    
    if not os.path.exists(folder1) or not os.path.exists(folder2):
        print("One or both folders do not exist")
        sys.exit(1)
        
    # Get sorted list of files from both folders
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith(('.png', '.jpg', '.jpeg'))])
    files2 = sorted([f for f in os.listdir(folder2) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(files1) != len(files2):
        print(f"Folders have different number of images: {len(files1)} vs {len(files2)}")
        sys.exit(1)
        
    total_mse = 0
    total_lpips = 0
    total_pose = 0
    num_images = len(files1)
    
    for f1, f2 in zip(files1, files2):
        img1 = cv2.imread(os.path.join(folder1, f1))
        img2 = cv2.imread(os.path.join(folder2, f2))
        
        if img1 is None or img2 is None:
            print(f"Error reading images: {f1} or {f2}")
            continue
            
        try:
            mse = calculate_mse(img1, img2)
            lpips_score = calculate_lpips(img1, img2)
            pose_score = compare_cube_poses(img1, img2)
            total_mse += mse
            total_lpips += lpips_score
            total_pose += pose_score
        except ValueError as e:
            print(f"Error processing {f1} and {f2}: {e}")
            continue
            
    if num_images > 0:
        avg_mse = total_mse / num_images
        avg_lpips = total_lpips / num_images
        avg_pose = total_pose / num_images
        # Weight pose comparison as 60% of final score
        total_score = 100 - (avg_mse + avg_lpips + avg_pose)
        print(f"Average MSE (0-20): {avg_mse:.2f}")
        print(f"Average LPIPS (0-20): {avg_lpips:.2f}")
        print(f"Average Pose Score (0-60): {avg_pose:.2f}")
        print(f"Total Score (0-20): {total_score:.2f}")
    else:
        print("No valid image pairs found")

if __name__ == "__main__":
    main()
