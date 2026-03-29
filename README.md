# Image Colorization using Deep Learning

This project uses a deep learning approach (Convolutional Neural Networks) to automatically convert grayscale images into realistic colored images. It leverages the **U-Net** architecture and the **LAB color space** for efficient and stable training.

---

## 🚀 **Introduction**

Image colorization is a classic computer vision problem. Traditionally, it required intensive manual effort. Modern AI can automate this process by learning how objects (like grass, sky, or skin) should look based on their texture and context in millions of example images.

## 🧠 **Core Concepts**

### 1. LAB Color Space
- **L (Luminance)**: Represents the brightness (the grayscale image).
- **a & b (Chrominance)**: Represent the color components.
The model takes the **L** channel as input and predicts the **a** and **b** channels.

### 2. U-Net Architecture
- **Encoder**: Compresses the image to extract high-level features.
- **Decoder**: Upsamples and reconstructs the color channels using skip connections to preserve spatial details.

---

## 🛠️ **Installation**

1. Ensure you have Python 3.8+ installed.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 🔄 **Usage Guide**

### 1. Preparing Training Data
Store your colored images in a folder (e.g., `dataset/`). The model will automatically convert them to grayscale for training.

### 2. Training the Model
Run the following command to start training:
```bash
python train.py --data_dir "path/to/your/images" --epochs 50 --batch_size 16
```

### 3. Colorizing a New Image
After training, or to test with an existing model:
```bash
python colorize.py --image "input_image.jpg" --model "colorization_model.h5"
```

### 4. Running the Web Dashboard (Frontend)
For a user-friendly interface, you can run the Streamlit app:
```bash
streamlit run app.py
```
This will open a browser window where you can upload images or capture photos via webcam!

---

## 🏗️ **Project Structure**

- `app.py`: Streamlit-based web dashboard.
- `model.py`: Defines the U-Net architecture.
- `utils.py`: Contains image processing and LAB conversion logic.
- `train.py`: Script for training the model.
- `colorize.py`: CLI script for colorizing B&W images.
- `requirements.txt`: List of dependencies including Streamlit.
- `generate_data.py`: Utility for synthetic data generation.

---

## 📈 **Next Steps**

- **GANs**: Use Generative Adversarial Networks for even more vibrant and realistic colors.
- **Video Colorization**: Extend the model to process video frames while maintaining temporal consistency.
- **Web Deployment**: Create a Streamlit or Flask app for user-friendly interaction.

---

## 🏁 **Conclusion**

AI is bringing history to life! By automating the colorization process, we can restore old photographs and films with unprecedented speed and accuracy.
