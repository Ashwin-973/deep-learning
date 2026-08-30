# CS3807 – Deep Learning Laboratory

## Experiment 5: Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation

This repository contains the implementation and report for **CS3807 – Deep Learning Laboratory, Experiment 5**.

The experiment studies the effect of:

* Weight initialization
* Regularization
* Batch Normalization
* Optimization algorithms
* CNN hyperparameters
* Transfer learning
* Fine-tuning
* 5-fold cross-validation
* Final model evaluation

The experiment uses **MobileNetV2** and the **Oxford-IIIT Pet Dataset** as specified in the laboratory experiment sheet. The images are resized to `224 × 224 × 3`.

---

## 1. Project Structure

A typical project directory should look like this:

```text
CS3807_Experiment_5/
│
├── experiment5.ipynb
├── CS3807_Experiment_5_Report.tex
├── README.md
│
├── initialization_accuracy.png
├── initialization_loss.png
├── regularization_accuracy.png
├── optimizer_accuracy.png
├── learning_rate.png
├── batch_size.png
├── dropout.png
├── transfer_learning.png
├── cv_accuracy.png
├── confusion_matrix.png
├── mobilenet_architecture.png
│
├── cross_validation_results.csv
├── final_model_results.csv
│
└── mobilenetv2_final_model.keras
```

The exact generated files may vary depending on how the notebook is executed.

---

# 2. Requirements

The project requires **Python 3.10 or newer**.

A GPU is recommended because several CNN training experiments and the 5-fold cross-validation stage can take significant time on a CPU.

### Main dependencies

The following Python packages are used:

```text
tensorflow
tensorflow-datasets
numpy
pandas
matplotlib
scikit-learn
```

The LaTeX report additionally requires a LaTeX distribution such as:

* TeX Live
* MiKTeX
* Overleaf

---

# 3. Installing the Dependencies

## Option 1 – Using pip

Create a virtual environment first:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Then install the required packages:

```bash
pip install tensorflow tensorflow-datasets numpy pandas matplotlib scikit-learn
```

To verify the installation:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

# 4. Google Colab

Google Colab is recommended if a local GPU is not available.

Open the notebook:

```text
experiment5.ipynb
```

Then select:

```text
Runtime → Change runtime type → T4 GPU
```

or another available GPU runtime.

Run the cells from top to bottom.

The notebook automatically downloads the Oxford-IIIT Pet Dataset through TensorFlow Datasets.

---

# 5. Dataset

The experiment uses the:

**Oxford-IIIT Pet Dataset**

The dataset contains images of cats and dogs belonging to **37 breeds**. The original images have different spatial dimensions, so they are resized to:

```text
224 × 224 × 3
```

The experiment maintains separate training, validation and test data. The independent test set is kept untouched during hyperparameter selection, as required by the experiment specification.

The dataset is loaded using:

```python
tfds.load("oxford_iiit_pet")
```

No manual dataset download is required when using the provided notebook.

---

# 6. Running the Notebook

Start Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Open:

```text
experiment5.ipynb
```

Run the cells in the following order.

### Step 1 – Import libraries

The first cells import TensorFlow, TensorFlow Datasets, NumPy, Pandas, Matplotlib and Scikit-learn.

### Step 2 – Configure the experiment

The notebook sets:

```text
Image size = 224 × 224
Number of classes = 37
Batch size = 32
Random seed = 42
```

### Step 3 – Load the dataset

The Oxford-IIIT Pet Dataset is loaded using TensorFlow Datasets.

### Step 4 – Preprocess the images

Images are resized to:

```text
224 × 224 × 3
```

and normalized/preprocessed for MobileNetV2.

### Step 5 – Apply data augmentation

The training data uses:

* Random horizontal flipping
* Random rotation
* Random zoom

### Step 6 – Train the baseline MobileNetV2

The baseline model uses ImageNet-pretrained MobileNetV2 with the pretrained base frozen and a new classification layer.

### Step 7 – Weight initialization study

The experiment compares:

```text
Zero
Random
Xavier / Glorot
He
```

The laboratory specification requires training-loss and validation-accuracy comparisons for these initialization strategies.

### Step 8 – Regularization study

The notebook compares:

```text
No Regularization
L2 Regularization
Dropout
Batch Normalization
```

Training and validation curves are generated to study overfitting and generalization.

### Step 9 – Batch Normalization example

The numerical Batch Normalization example from the experiment is reproduced using:

```text
x = [2, 4, 6, 8]
```

The calculated mean is:

```text
5
```

and the variance is:

```text
5
```

The normalized values are approximately:

```text
[-1.342, -0.447, 0.447, 1.342]
```

### Step 10 – Optimizer comparison

The notebook compares:

```text
SGD
Momentum
RMSProp
Adam
```

The experiment evaluates convergence behaviour, training loss and validation accuracy.

### Step 11 – Hyperparameter tuning

The notebook investigates:

```text
Learning Rate:
    0.001
    0.0001

Batch Size:
    16
    32
    64

Dropout:
    0
    0.25
    0.5
```

The experiment recommends changing one hyperparameter at a time while keeping the remaining settings fixed during a controlled study.

### Step 12 – Transfer learning

MobileNetV2 pretrained on ImageNet is used.

Two approaches are evaluated:

```text
Feature Extraction
Fine-Tuning
```

Feature extraction freezes the pretrained base.

Fine-tuning unfreezes selected upper layers and uses a smaller learning rate.

### Step 13 – 5-fold cross-validation

Four promising configurations are selected and evaluated using:

```text
5-fold cross-validation
```

The mean and standard deviation are calculated for each configuration.

The experiment specifies reporting cross-validation performance as:

```text
Mean ± Standard Deviation
```

The independent test set must not be used during this stage.

### Step 14 – Final evaluation

After selecting the best configuration, the model is evaluated using the independent test set.

The following metrics are calculated:

```text
Test Accuracy
Precision
Recall
F1-score
Training Time
Number of Parameters
```

A confusion matrix is also generated.

---

# 7. Important Note About Training Time

The experiment contains several independent training runs followed by 5-fold
cross-validation.

Therefore, running the complete notebook can take considerably longer than
training a single CNN.

For initial testing, it is reasonable to use fewer epochs to verify that all
cells work correctly.

For the final experiment, use the intended epoch count and record the actual
results generated by the run.

---

# 8. Generated Outputs

Running the notebook produces several plots.

### Weight Initialization

```text
initialization_loss.png
initialization_accuracy.png
```

These compare training loss and validation accuracy for the different
initialization methods.

### Regularization

```text
regularization_accuracy.png
```

This compares the validation performance of the regularization methods.

### Optimization

```text
optimizer_accuracy.png
```

This compares SGD, Momentum, RMSProp and Adam.

### Hyperparameter Tuning

```text
learning_rate.png
batch_size.png
dropout.png
```

These show the effect of the selected hyperparameters on validation accuracy.

### Transfer Learning

```text
transfer_learning.png
```

This compares feature extraction and fine-tuning.

### Cross-Validation

```text
cv_accuracy.png
```

This shows the mean validation accuracy of the selected configurations with
standard-deviation error bars.

### Final Evaluation

```text
confusion_matrix.png
```

This shows the classification performance of the final model.

### Architecture

```text
mobilenet_architecture.png
```

This provides a simplified conceptual representation of MobileNetV2.

---

# 9. Saved Results

The notebook can save the experimental results as:

```text
cross_validation_results.csv
```

and:

```text
final_model_results.csv
```

The trained final model can be saved as:

```text
mobilenetv2_final_model.keras
```

---

# 10. Compiling the LaTeX Report

The report is provided as:

```text
CS3807_Experiment_5_Report.tex
```

Make sure all the `.png` files referenced by the `.tex` file are in the same
directory as the LaTeX source.

## Using pdflatex

Run:

```bash
pdflatex CS3807_Experiment_5_Report.tex
```

For the table of contents and references to update correctly, run it again:

```bash
pdflatex CS3807_Experiment_5_Report.tex
```

This generates:

```text
CS3807_Experiment_5_Report.pdf
```

---

# 11. Using Overleaf

The easiest option if LaTeX is not installed locally is Overleaf.

Upload:

```text
CS3807_Experiment_5_Report.tex
```

along with all the generated `.png` files.

Set the compiler to:

```text
pdfLaTeX
```

Then compile the project.

---

# 12. Reproducibility

A random seed of:

```python
SEED = 42
```

is used for NumPy and TensorFlow.

However, exact training results can still vary slightly depending on:

* TensorFlow version
* CUDA version
* GPU hardware
* CPU hardware
* available system resources
* numerical operations
* training execution environment

Therefore, small differences in accuracy or training time between machines
are expected.

---

# 13. Expected Experimental Workflow

The overall workflow is:

```text
Oxford-IIIT Pet Dataset
          |
          v
   Image Preprocessing
          |
          v
      MobileNetV2
          |
          +--------------------+
          |                    |
          v                    v
 Weight Initialization    Regularization
          |                    |
          +---------+----------+
                    |
                    v
             Optimizer Study
                    |
                    v
          Hyperparameter Tuning
                    |
                    v
            Transfer Learning
                    |
                    v
              Fine-Tuning
                    |
                    v
        Select 3–4 Configurations
                    |
                    v
          5-Fold Cross-Validation
                    |
                    v
             Select Best Model
                    |
                    v
          Independent Test Set
                    |
                    v
        Accuracy / Precision /
        Recall / F1 / Confusion
              Matrix
```

---

# 14. Main Dependencies

| Package               | Purpose                                 |
| --------------------- | --------------------------------------- |
| `tensorflow`          | CNN construction and training           |
| `tensorflow-datasets` | Oxford-IIIT Pet Dataset loading         |
| `numpy`               | Numerical operations                    |
| `pandas`              | Result tables and CSV files             |
| `matplotlib`          | Training curves and plots               |
| `scikit-learn`        | Cross-validation and evaluation metrics |
| `jupyter`             | Running the notebook                    |

Install everything with:

```bash
pip install tensorflow tensorflow-datasets numpy pandas matplotlib scikit-learn jupyter
```

---

# 15. References

The experiment is based on the references listed in the laboratory document:

1. Ian Goodfellow, Yoshua Bengio and Aaron Courville, *Deep Learning*, MIT Press, 2016.
2. Sergey Ioffe and Christian Szegedy, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, ICML, 2015.
3. Mark Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR, 2018.
4. Omkar M. Parkhi et al., *Cats and Dogs*, CVPR, 2012.
5. TensorFlow Documentation.
6. Keras Documentation.

The laboratory sheet also specifies MobileNetV2 as the CNN architecture and
the Oxford-IIIT Pet Dataset as the experimental dataset.

---

## 16. Quick Start

For the shortest setup:

```bash
git clone <repository-url>
cd CS3807_Experiment_5

python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
# venv\Scripts\activate

pip install tensorflow tensorflow-datasets numpy pandas matplotlib scikit-learn jupyter

jupyter notebook
```

Open:

```text
experiment5.ipynb
```

and run the notebook from top to bottom.

After completing the experiment, compile:

```text
CS3807_Experiment_5_Report.tex
```

to generate the final PDF report.
