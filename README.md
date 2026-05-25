# 🏨 Hotel Price Prediction System

A machine learning-powered hotel price prediction system for Indian cities. Predicts room prices using **Linear Regression**, **Random Forest**, and **Gradient Boosting** models, and recommends the top 5 hotels based on your criteria.

---

## ✨ Features

- **3 ML Models**: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor
- **Indian Cities**: 25+ cities including Mumbai, Delhi, Bangalore, Goa, Jaipur, Udaipur, etc.
- **Smart Predictions**: Based on location, season, guests, room type, weekday/weekend, and local events
- **Top 5 Recommendations**: Get the best hotel picks with ratings and pricing
- **Dark Glassmorphism UI**: Professional, modern, and responsive web interface
- **Model Comparison**: See predictions from each model side by side
- **Rule-Based Fallback**: Works even before training the ML models

---

## 📁 Project Structure

```
ML_LabProject/
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Data loading, EDA, preprocessing
│   ├── 02_linear_regression.ipynb     # Linear Regression training
│   ├── 03_random_forest.ipynb         # Random Forest training
│   └── 04_gradient_boosting.ipynb     # Gradient Boosting training
├── models/                            # Saved trained models (.pkl files)
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
├── static/
│   ├── css/style.css                  # Dark glassmorphism theme
│   └── js/main.js                     # Frontend JavaScript
├── templates/
│   └── index.html                     # Main UI template
├── app.py                             # Flask server
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Google Colab account (for training notebooks)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models (Google Colab)

Open each notebook in **Google Colab** in order:

1. `notebooks/01_data_exploration.ipynb` — Loads and preprocesses the dataset
2. `notebooks/02_linear_regression.ipynb` — Trains Linear Regression model
3. `notebooks/03_random_forest.ipynb` — Trains Random Forest model
4. `notebooks/04_gradient_boosting.ipynb` — Trains Gradient Boosting model

Each notebook will:
- Import the dataset directly from Kaggle (no local download needed)
- Train the model and evaluate performance
- Save the trained model as a `.pkl` file to `models/`

> **Note**: Download the `.pkl` files from Colab and place them in the `models/` directory.

### 3. Run the Flask Server

```bash
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5000**

> **Note**: The app works even without trained models — it uses an intelligent rule-based fallback system for predictions.

---

## 📊 Input Features

| Feature | Description | Values |
|---------|-------------|--------|
| **City** | Indian city location | Mumbai, Delhi, Bangalore, Goa, etc. |
| **Season** | Time of year | Summer, Winter, Monsoon, Spring, Autumn |
| **Guests** | Number of guests | 1 – 10 |
| **Room Type** | Type of room | Standard, Deluxe, Suite, Premium, Executive |
| **Day Type** | Weekday or Weekend | Toggle switch |
| **Local Events** | Events happening nearby | Toggle switch |

---

## 🤖 Algorithms Used

### 1. Linear Regression
- Baseline model for price prediction
- Fast training and inference
- Best for understanding feature importance

### 2. Random Forest Regressor
- Ensemble of decision trees
- Handles non-linear relationships well
- Robust to outliers

### 3. Gradient Boosting Regressor
- Sequential ensemble method
- Often achieves highest accuracy
- Uses learning rate and max depth tuning

---

## 📈 Output

- **Predicted Room Price**: Ensemble average of all 3 models (₹)
- **Model Comparison**: Individual predictions from each algorithm
- **Top 5 Hotels**: Recommended hotels with ratings, prices, and details

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Flask (Python) |
| **ML Models** | scikit-learn |
| **Data Processing** | pandas, NumPy |
| **Frontend** | HTML5, CSS3, JavaScript |
| **UI Theme** | Custom Dark Glassmorphism |
| **Font** | Inter (Google Fonts) |
| **Model Serialization** | joblib |

---

## 📝 License

This project is for educational purposes — ML Lab Project © 2026.
