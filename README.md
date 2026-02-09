# Linear Regression Using Python

A clean, minimal implementation of **Linear Regression from scratch** using NumPy. This project is built for learning and quick experimentation, with ready-to-run notebooks and sample data.

---

## ✨ Highlights

- Pure NumPy implementation (no scikit-learn)
- Simple API with `LinearModel` class
- Example notebooks included
- Works with your own CSV/Excel datasets

---

## 🚀 Quick Start

1. **Clone the repo**

2. **Install dependencies**

```bash
pip install numpy pandas matplotlib xlrd
```

3. **Run a notebook**

Open and run:
- `example_implementation.ipynb`
- `main.ipynb`

---

## ✅ Basic Usage

```python
from linear_model import LinearModel
import numpy as np

# X shape: (m, n) and y shape: (m, 1)
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([[1.2], [1.9], [3.1], [3.9]])

model = LinearModel(num_features=1)
losses = model.train(X, y, iterations=2000, lr=0.01)

y_pred = model.forward_pass(X)
print(y_pred)
```

---

## 📊 Using the Sample Data

Sample files are included:
- `chirps.csv`
- `chirps.xls`

You can load the Excel file like this:

```python
import pandas as pd

df = pd.read_excel("chirps.xls", engine="xlrd")
```

---

## 📁 Project Structure

```
.
├── linear_model.py                 # Core Linear Regression implementation
├── example_implementation.ipynb    # Step-by-step walkthrough
├── main.ipynb                       # Main usage notebook
├── chirps.csv                       # Sample dataset (CSV)
├── chirps.xls                       # Sample dataset (Excel)
└── README.md                        # Project guide
```

---

## 🧠 How the Model Works

The model minimizes Mean Squared Error using gradient descent:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
$$

Parameters are updated iteratively using:

$$
\theta := \theta - \alpha \nabla J(\theta)
$$

---

## 🔧 Tips

- Scale your features if values are large.
- Increase `iterations` for better convergence.
- Tune `lr` (learning rate) if loss is unstable.

---

## 📬 Credits

Created by **Rehnoor Aulakh**.

If you find this helpful, feel free to star the repo and share it.
