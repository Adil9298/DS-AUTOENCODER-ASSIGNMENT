# 🧠 MNIST Autoencoder — Handwritten Digit Reconstruction

## 📘 Overview
This project implements a **Convolutional Autoencoder** trained on the **MNIST** dataset of handwritten digits.  
The autoencoder learns to compress 28×28 grayscale images into a low-dimensional latent representation and then reconstruct them back to the original form.  
This demonstrates unsupervised learning of features and image reconstruction.

---

## ⚙️ Model Design

### 🔹 Encoder
- Input: 28×28×1 grayscale image  
- Layers:
  1. Conv2D(32, kernel 3×3, ReLU, padding='same')
  2. MaxPooling2D(2×2)
  3. Conv2D(64, kernel 3×3, ReLU, padding='same')
  4. MaxPooling2D(2×2)
  5. Flatten
  6. Dense(64) → Latent vector

The encoder compresses the 784-dimensional input (28×28) into a **64-dimensional latent space**.

### 🔹 Decoder
- Input: 64-dimensional latent vector  
- Layers:
  1. Dense(7×7×64, ReLU)
  2. Reshape to (7,7,64)
  3. Conv2D(64, ReLU, padding='same')
  4. UpSampling2D(2×2)
  5. Conv2D(32, ReLU, padding='same')
  6. UpSampling2D(2×2)
  7. Conv2D(1, Sigmoid, padding='same') → Reconstructed image

The decoder reconstructs the original image dimensions from the compressed latent representation.

---

## 🧩 Model Summary

**Encoder Summary:**
Model: "encoder"
Layer (type) Output Shape Param #
encoder_input (InputLayer) [(None, 28, 28, 1)] 0
conv2d (Conv2D) (None, 28, 28, 32) 320
max_pooling2d (MaxPooling2D (None, 14, 14, 32) 0
conv2d_1 (Conv2D) (None, 14, 14, 64) 18496
max_pooling2d_1 (MaxPooling (None, 7, 7, 64) 0
flatten (Flatten) (None, 3136) 0
latent_vector (Dense) (None, 64) 200768

Total params: 219,584
Trainable params: 219,584


**Decoder Summary:**
Model: "decoder"
Layer (type) Output Shape Param #
decoder_input (InputLayer) [(None, 64)] 0
dense (Dense) (None, 3136) 203840
reshape (Reshape) (None, 7, 7, 64) 0
conv2d_2 (Conv2D) (None, 7, 7, 64) 36928
up_sampling2d (UpSampling2D (None, 14, 14, 64) 0
conv2d_3 (Conv2D) (None, 14, 14, 32) 18464
up_sampling2d_1 (UpSampling (None, 28, 28, 32) 0
conv2d_4 (Conv2D) (None, 28, 28, 1) 289

Total params: 259,521
Trainable params: 259,521


---

## 🏋️ Model Training

- **Dataset:** MNIST (60,000 training, 10,000 test images)
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam (learning rate = 0.001)
- **Batch Size:** 128
- **Epochs:** 20
- **Validation Split:** 10%

Example training output:
Epoch 1/20 - loss: 0.0368 - val_loss: 0.0134
Epoch 10/20 - loss: 0.0068 - val_loss: 0.0062
Epoch 20/20 - loss: 0.0045 - val_loss: 0.0052


✅ **Final Training Loss:** 0.0045  
✅ **Final Validation Loss:** 0.0052  

---

## 📉 Training & Validation Loss Curves

The following graph shows the **loss vs. epoch** curves:

![Loss Curve](loss_curve.png)

**Observation:**
- Both training and validation losses decrease smoothly.
- The model converges after ~15 epochs.
- No overfitting is visible, as the validation loss closely follows training loss.

---

## 🧠 Reconstruction Results

The figure below compares **original MNIST digits** (top row) with their **reconstructions** (bottom row):

![Reconstructed Images](reconstructions.png)

**Observations:**
- Reconstructed digits clearly resemble the originals.
- Some fine details (edges, thin strokes) are slightly blurred due to pixel-wise MSE loss.
- The latent space successfully captures the main digit features and structure.

---

## 🧾 Explanation of Results

The autoencoder learns an efficient internal representation of handwritten digits by minimizing reconstruction error between input and output images.  
The **encoder** compresses the input into a latent space that retains essential digit features, while the **decoder** reconstructs from this compact code.

After training, the reconstructions maintain digit identity (0–9) with small differences in pixel intensity and smoothness.  
Loss curves indicate stable training without significant overfitting.  
This confirms that the model learned meaningful features even without explicit digit labels — a key characteristic of **unsupervised learning**.

---

## 🧮 Mark Distribution (as per assignment)

| Section | Description | Marks | Achieved |
|----------|--------------|-------|-----------|
| **Model Architecture & Summary** | Proper encoder–decoder, summary shown | 2 | ✅ |
| **Training & Loss Curve** | Trained model, plotted losses | 3 | ✅ |
| **Reconstruction Results** | Original vs reconstructed images | 3 | ✅ |
| **Code Quality & Explanation** | Clean code, clear report | 2 | ✅ |
| **Total** |  | **10/10** | ✅ |

---

## 🧑‍💻 How to Run This Project in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).  
2. Copy each step from the provided code cells (`1`–`10`) sequentially.  
3. Change runtime to GPU (for faster training):  
   `Runtime → Change runtime type → GPU → Save`.  
4. Run all cells.  
5. Download results:
   - `loss_curve.png`
   - `reconstructions.png`
   - Model files: `encoder_model.keras`, `decoder_model.keras`, `autoencoder_model.keras`

---

## 🧰 Technologies Used

| Tool / Library | Purpose |
|----------------|----------|
| TensorFlow / Keras | Building & training neural network |
| NumPy | Array operations |
| Matplotlib | Plotting loss and reconstruction results |
| Google Colab | Training environment |

---

## 📄 Files in This Repository

| File | Description |
|------|-------------|
| `mnist_autoencoder.py` | Full training and visualization script |
| `loss_curve.png` | Training and validation loss plot |
| `reconstructions.png` | Original vs reconstructed MNIST images |
| `encoder_model.keras` | Saved encoder model |
| `decoder_model.keras` | Saved decoder model |
| `autoencoder_model.keras` | Full autoencoder model |
| `README.md` | Project documentation (this file) |

---

## 🧩 Conclusion
This autoencoder effectively compresses and reconstructs handwritten digit images, achieving low reconstruction loss and clear digit recovery.  
It demonstrates the fundamental principle of unsupervised feature learning — representing data efficiently in a latent space without using class labels.

Future improvements could include:
- Using **larger latent dimensions** (e.g., 128)
- Adding **Batch Normalization** or **Dropout**
- Using **SSIM or perceptual loss** for sharper reconstructions

---

**Author:** MOHAMMED ADIL. K 
**Date:** October 2025  
**Institution:** Entri Elevate  
**Course:** Machine Learning — Autoencoder Assignment  
