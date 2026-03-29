import numpy as np
import cv2
import os

def generate_synthetic_data(dir_name='synthetic_data', num_images=10):
    """
    Generates synthetic images for testing the colorization model.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    for i in range(num_images):
        # Create a white background
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        
        # Draw some random shapes with different colors
        # 1. Blue circle
        cv2.circle(img, (np.random.randint(50, 200), np.random.randint(50, 200)), 
                   np.random.randint(20, 50), (255, 0, 0), -1)
        
        # 2. Green rectangle
        cv2.rectangle(img, (np.random.randint(50, 150), np.random.randint(50, 150)), 
                      (np.random.randint(150, 250), np.random.randint(150, 250)), (0, 255, 0), -1)
        
        # 3. Red triangle
        pts = np.array([[np.random.randint(0, 256), np.random.randint(0, 256)], 
                        [np.random.randint(0, 256), np.random.randint(0, 256)], 
                        [np.random.randint(0, 256), np.random.randint(0, 256)]], np.int32)
        cv2.fillPoly(img, [pts], (0, 0, 255))
        
        # Save the image
        cv2.imwrite(os.path.join(dir_name, f'img_{i}.jpg'), img)
    
    print(f"Generated {num_images} synthetic images in {dir_name}/")

if __name__ == "__main__":
    generate_synthetic_data()
