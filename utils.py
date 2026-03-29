import numpy as np
import cv2
import os
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def preprocess_image(image_path, size=(256, 256)):
    """
    Loads an image, converts to Scientific Lab.
    Returns L (repeated 3 times) and AB channels.
    """
    try:
        img = load_img(image_path, target_size=size)
        img_array = img_to_array(img) / 255.0
        lab_img = rgb2lab(img_array)
        
        L = lab_img[:, :, 0] / 100.0
        AB = lab_img[:, :, 1:] / 128.0
        
        # 3-channel L for Transfer Learning
        L_3ch = np.repeat(L[:, :, np.newaxis], 3, axis=2)
        return L_3ch, AB
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def preprocess_array(img_array, size=(256, 256)):
    """
    Takes an RGB array, converts to Scientific Lab.
    Returns L (repeated 3 times) and AB channels.
    """
    img_resized = cv2.resize(img_array, size)
    img_normalized = img_resized / 255.0
    lab_img = rgb2lab(img_normalized)
    
    L = lab_img[:, :, 0] / 100.0
    AB = lab_img[:, :, 1:] / 128.0
    
    L_3ch = np.repeat(L[:, :, np.newaxis], 3, axis=2)
    return L_3ch, AB

def postprocess_image(L_3ch, AB, size=(256, 256)):
    """
    Combines 3-ch L and AB channels back to RGB.
    """
    L = L_3ch[:, :, 0] * 100.0
    AB = AB * 128.0
    
    lab_img = np.zeros((size[0], size[1], 3))
    lab_img[:, :, 0] = L.reshape(size[0], size[1])
    lab_img[:, :, 1:] = AB
    
    rgb_img = lab2rgb(lab_img)
    return (rgb_img * 255).astype(np.uint8)

def data_generator(image_dir, batch_size=32, target_size=(256, 256)):
    """
    Generator for training with Transfer Learning input shapes.
    """
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    while True:
        np.random.shuffle(image_files)
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            L_batch = []
            AB_batch = []
            for f in batch_files:
                L, AB = preprocess_image(os.path.join(image_dir, f), size=target_size)
                if L is not None:
                    L_batch.append(L)
                    AB_batch.append(AB)
            
            if L_batch:
                yield np.array(L_batch), np.array(AB_batch)

def color_pro_image(img_array):
    """
    OpenCV DNN Pre-trained Model (Zhang et al.)
    """
    proto = "models/colorization_deploy_v2.prototxt"
    model = "models/colorization_release_v2.caffemodel"
    hull = "models/pts_in_hull.npy"
    
    if not os.path.exists(proto) or not os.path.exists(model) or not os.path.exists(hull):
        return None, "Model files not found. Please wait for download."
    
    net = cv2.dnn.readNetFromCaffe(proto, model)
    pts = np.load(hull)
    
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    img_rgb = img_array.astype("float32") / 255.0
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    
    resized_l = cv2.resize(lab[:, :, 0], (224, 224), interpolation=cv2.INTER_AREA)
    resized_l -= 50
    
    net.setInput(cv2.dnn.blobFromImage(resized_l))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img_array.shape[1], img_array.shape[0]))
    
    L = lab[:, :, 0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_Lab2RGB)
    colorized = np.clip(colorized, 0, 1)
    
    return (colorized * 255).astype("uint8"), None

print("Utils module loaded successfully.")
