# AeroFare — Flight Price Prediction

A machine learning web application that predicts Indian domestic flight ticket prices using a Decision Tree Regressor trained on 300,000+ real flight records.

## 🚀 Live Demo
👉 https://flight-app-ktkn.onrender.com

## 📌 Features
- Predict flight fare instantly based on airline, route, class, stops, duration and booking window
- Price trend chart showing how fare changes across 49 days before departure
- Cheapest day finder — highlights the best day to book
- Glassmorphism UI with real airplane photo background
- Fully responsive on mobile and desktop

## 🛠️ Tech Stack
- **Python 3.11** — core language
- **scikit-learn** — Decision Tree model, ColumnTransformer, cross validation
- **Pandas & NumPy** — data processing
- **Flask** — web framework
- **pickle** — model serialisation
- **Chart.js** — price trend chart
- **Render.com** — deployment

## 📊 Model Performance
| Model | R² Score | MAE |
|---|---|---|
| Linear Regression | 0.910 | ₹4,488 |
| Decision Tree ✓ | 0.961 | ₹2,479 |
| Random Forest | 0.986 | ₹1,064 |

Cross Validation (5-fold): Mean R² = 0.958, Std Dev = 0.001

## 📁 Project Structure
