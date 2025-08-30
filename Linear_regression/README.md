
---

# FCFNN: Approximating $f(x)=5x^2+10x-2$

This project builds and trains a **Fully Connected Feedforward Neural Network (FCFNN)** to learn the quadratic function

$$
f(x) = 5x^2 + 10x - 2
$$

from synthetic data. After training, it evaluates on validation/test splits and plots **one figure with two subplots**: (left) the **original** curve and (right) the **predicted** curve.

## ✨ What this code does

* Generates $N$ random inputs $x \sim \mathcal{U}[-100,100]$ and computes targets $y=f(x)$.
* Splits data into **70% train**, **10% validation**, **20% test**:

  * First split: 70% train, 30% temporary set.
  * Second split: temporary 30% → **validation\:test = 1:2** (i.e., `test_size=2/3`) ⇒ overall 10%/20%.
* Builds a compact MLP: `Input(1) → Dense(32, ReLU) → Dense(32, ReLU) → Dense(1, linear)`.
* Trains with **Adam** optimizer and **MSE** loss.
* Evaluates **Validation MSE** (10%) and **Test MSE** (20%).
* Plots a single figure with **two subplots**:

  * Left: Original $f(x)$
  * Right: Predicted $\hat f(x)$ from the trained model

## 🧰 Requirements

* Python 3.8+
* TensorFlow 2.x
* NumPy
* Matplotlib
* scikit-learn

Install:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

> 💡 On systems without a GPU, `tensorflow` (CPU) is fine. For NVIDIA GPUs, use the GPU-enabled TensorFlow that matches your CUDA/cuDNN setup.

## ▶️ How to run

Save the script as `main.py`, then:

```bash
python main.py
```

You’ll see console output like:

```
Train: 1400 | Val: 200 | Test: 400
Validation MSE (10%): 0.0000
Test MSE (20%): 0.0000
```

…and a window with **one figure** containing two subplots:

* **Original y = f(x)** (left)
* **Predicted y (FCFNN)** (right)

## 📁 File structure (suggested)

```
.
├── main.py         # the script
└── README.md       # this file
```

## 🔧 Customize

* **Dataset size**: change `data_generate(n=...)`.
* **Randomness**: set `random_seed`.
* **Network depth/width**: edit `build_model()` (layers/units/activations).
* **Training**: modify `epochs`, `batch_size`, optimizer, loss.
* **Input range**: adjust `np.random.uniform(-100, 100, ...)`.
* **Grid density for plot**: change `np.linspace(..., 600)`.

## ❓ FAQ

**Q: Why `test_size=2/3` in the second split?**
A: After taking 70% for training, 30% remains. We want overall **10% validation** and **20% test**.
30% × **1/3** = 10% (val) and 30% × **2/3** = 20% (test). Hence `test_size=2/3`.

**Q: What does Adam do?**
A: It computes gradients and adapts per-parameter learning rates using running means/variances of gradients, giving fast and stable training.

## 🧪 Expected outcome

Because the data are noise-free and the target is a smooth quadratic, the network should fit very closely. The predicted curve typically **overlaps** the true curve across the sampled domain, and both Validation/Test MSE should be **near zero**.

## 🛠 Troubleshooting

* `ModuleNotFoundError: No module named 'tensorflow'`
  → `pip install tensorflow`
* Plot window doesn’t show
  → In notebooks, add `%matplotlib inline`; in headless servers, save the figure via `plt.savefig(...)`.
* Very large losses
  → Ensure you didn’t accidentally change the function, range, or target dtype.

## 📎 References

* TensorFlow Keras API: [https://www.tensorflow.org/api\_docs/python/tf/keras](https://www.tensorflow.org/api_docs/python/tf/keras)
* Adam optimizer paper: *Kingma & Ba (2015)*

---


