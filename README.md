# Oxford Research Fire Detection System

A **fire detection machine learning model** that achieves an **85% accuracy**, capable of detecting fire in images. This project integrates both a **machine learning backend** (TensorFlow) and a **simple frontend** using Tkinter to provide a user-friendly interface for image upload and fire detection.

## Features
- **Machine Learning Model**: A neural network built using TensorFlow and Keras to classify images as fire or no fire.
- **Image Augmentation**: Use of `ImageDataGenerator` for image augmentation (rescaling, shear, zoom, etc.).
- **Tkinter GUI**: A graphical interface allowing users to upload images and detect fire.
- **Callback Functions**: Early stopping callback to avoid overfitting.

## Dataset
Download the dataset from Kaggle: [Fire Detection Dataset](https://www.kaggle.com/datasets/atulyakumar98/test-dataset?resource=download)

## Certificate
This project was developed as part of the AIIP Internship. You can view the certificate [here](https://drive.google.com/file/d/1FYp2a5eVcd632fG2EXYOdb2HBX2PC0P_/view?usp=sharing).

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- Tkinter
- OpenCV
- PIL (Pillow)
- NumPy
- Matplotlib

To install the required libraries, run:

```bash
pip install tensorflow numpy matplotlib pandas pillow opencv-python
```

## File Structure
```
.
├── 1/                  # Folder containing images of fire
├── 0/                  # Folder containing images of no fire
├── fire_detection.py    # Backend for training and running the ML model
├── fire_detection_gui.py # Tkinter GUI for image upload and detection
└── README.md            # Project documentation
```

## How to Run the Project

### Backend (Training the Model)

1. Clone this repository:

```bash
git clone https://github.com/your-username/fire-detection-system.git
cd fire-detection-system
```

2. Download the dataset from the link above and place it in the respective directories (`1/` for fire images, `0/` for no fire images).
3. Run the backend code to train the model:

```bash
python fire_detection.py
```

### Frontend (Tkinter GUI)

Once the model is trained, you can run the GUI for fire detection:

```bash
python fire_detection_gui.py
```

1. The GUI will open with an option to upload an image.
2. After selecting an image, the model will predict whether it contains fire or not and display the result.

## Model Architecture

- **Input Layer**: Image rescaling and flattening.
- **Hidden Layers**: Two fully connected dense layers with 128 and 64 neurons, respectively.
- **Dropout**: Applied with a 20% rate to prevent overfitting.
- **Output Layer**: Softmax activation for binary classification (Fire, No Fire).

```python
Model_Two = tf.keras.models.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Flatten(input_shape=(256,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2, activation="softmax")
])
```

## Sample Output

After selecting an image:

- If fire is detected: The system will display a **red warning** "Fire Detected".
- If no fire is detected: The system will display a **green message** "No Fire Detected".
- If the image is invalid: It will prompt the user to select a different image.

## Future Enhancements
- Add a **real-time video** fire detection feature using webcam integration.
- Improve the accuracy of the model with more training data and advanced model architectures (e.g., CNNs).
- Deploy the model as a **web application** using frameworks like Flask or Streamlit.
