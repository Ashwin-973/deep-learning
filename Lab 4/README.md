# CS3807 – Deep Learning Laboratory
## Experiment 4 – Comparative Study of Deep CNN Architectures Using Transfer Learning

This repository contains the implementation, generated visualizations, and report material for **Experiment 4** of the CS3807 – Deep Learning Laboratory at Shiv Nadar University Chennai.

The experiment studies the evolution of Convolutional Neural Networks and demonstrates transfer learning and fine-tuning using a pretrained ResNet50 model.

---

## 1. Experiment Objective

The objectives of this experiment are:

- Study the evolution of deep CNN architectures.
- Compare LeNet-5, AlexNet, VGG16, GoogleNet, and ResNet.
- Understand transfer learning.
- Implement transfer learning using a pretrained CNN.
- Fine-tune selected convolutional layers.
- Evaluate classification performance using standard metrics.
- Analyze training and validation behavior using plots.

---

## 2. Model Used

The primary implementation uses:

**ResNet50 pretrained on ImageNet**

The original ImageNet classification layer is removed and replaced with a custom classifier:

```text
Input Image
     ↓
ResNet50 Convolutional Base
     ↓
Global Average Pooling
     ↓
Dense Layer (256 units, ReLU)
     ↓
Dropout (0.5)
     ↓
Dense Layer (10 units, Softmax)
     ↓
CIFAR-10 Prediction

---

## 3. Dependencies

The implementation requires Python 3.9 or newer.

Main dependencies:

TensorFlow
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
Install using pip
pip install tensorflow numpy matplotlib seaborn scikit-learn jupyter

---

## 4. Running the Experiment

Step 1 – Clone or download the repository
git clone <repository-url>
cd experiment-4

If the repository is already downloaded, simply navigate into the project directory:

cd experiment-4
Step 2 – Create a virtual environment

Creating a virtual environment is recommended.

Windows
python -m venv venv
venv\Scripts\activate
Linux/macOS
python3 -m venv venv
source venv/bin/activate
Step 3 – Install dependencies
pip install -r requirements.txt
Step 4 – Run the Jupyter Notebook

Start Jupyter:

jupyter notebook

Open:

experiment4_resnet50.ipynb

Run the cells sequentially.