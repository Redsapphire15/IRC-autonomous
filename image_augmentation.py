import cv2
import os
import random


# Define the path to the folder containing your images
folder_path = '/home/kavin/Downloads/arrow_distant'

# Make sure to replace these parameters with your desired augmentation options
rotation_angles = [-10, -5, 0, 5, 10]  # Rotation angles in degrees
scale_factors = [0.8, 0.9, 1.0, 1.1, 1.2]  # Scaling factors
flip_options = [0, 1]  # 0 for horizontal flip, 1 for vertical flip
brightness_factors = [-75, -45, -10, 10, 45, 75]  # Adjust brightness levels


# Loop through each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):  # Add more extensions if needed
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        # Apply augmentations
        for i in range(5):  # Generate 10 augmented images per original image
            augmented_img = img.copy()
            
            # Randomly choose augmentation parameters
            rotation_angle = random.choice(rotation_angles)
            scale_factor = random.choice(scale_factors)
            flip_option = random.choice(flip_options)
            brightness_factor = random.choice(brightness_factors)
            
            
            # Apply transformations
            rows, cols, _ = augmented_img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, scale_factor)
            augmented_img = cv2.warpAffine(augmented_img, M, (cols, rows))
            augmented_img = cv2.flip(augmented_img, flip_option)
            augmented_img = cv2.convertScaleAbs(augmented_img, alpha=1, beta=brightness_factor)
            
            
            # Save the augmented image
            new_filename = f"{filename.split('.')[0]}_{i+1}.{filename.split('.')[-1]}"
            cv2.imwrite(os.path.join(folder_path, new_filename), augmented_img)
