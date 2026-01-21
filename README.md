# 📊 Residential Real Estate Price Analysis — Bishkek

## 📌 Project Overview
This project presents an end-to-end data analysis and price prediction pipeline for residential real estate in **Bishkek**.  
The main goal of the project is to identify key factors influencing housing prices and to build machine learning models capable of accurately forecasting property values.

The project covers the complete data science workflow: data collection, data cleaning, exploratory data analysis, feature engineering, modeling, and evaluation.

## 📊 Data Source
```https://www.kaggle.com/competitions/predicting-the-cost-of-apartments/```

---

## 🛠️ Tech Stack
- Python  
- pandas, NumPy  
- scikit-learn  
- matplotlib, seaborn  
- XGBoost, Random Forest, Linear Regressions L1 and L2


---

## 📄 Dataset Description

The dataset contains detailed information about residential real estate listings.  
Each row represents a single property listing, including location data, pricing, technical characteristics, and user engagement metrics.

### 🏷️ General Information
- **header_details** — Listing title or short description
- **address** — Property address
- **latitude** — Latitude coordinate
- **longitude** — Longitude coordinate
- **user_name** — Name of the listing owner or agent
- **user_url** — URL to the user’s profile
- **tel_number / Телефон** — Contact phone number

### 💰 Pricing & Listing Metadata
- **price_dollars** — Property price in USD
- **Тип предложения** — Type of offer (sale / rent)
- **publicated** — Publication date of the listing
- **upped** — Date of last listing update
- **views** — Number of views
- **hearts** — Number of likes or favorites
- **num_of_comments** — Number of comments
- **pictures** — Number of images in the listing

### 🏠 Property Characteristics
- **Серия** — Building series
- **Дом** — Building type
- **Этаж** — Apartment floor
- **Площадь** — Total area (m²)
- **Площадь участка** — Land plot area (for houses)
- **Высота потолков** — Ceiling height
- **Состояние** — Property condition
- **Санузел** — Bathroom type
- **Балкон** — Balcony availability
- **Мебель** — Furniture availability
- **Пол** — Floor material
- **Входная дверь** — Entrance door type
- **Парковка** — Parking availability
- **Безопасность** — Security features

### 🔥 Utilities & Infrastructure
- **Отопление** — Heating type
- **Газ** — Gas availability
- **Электричество** — Electricity availability
- **Вода / Питьевая вода** — Drinking water access
- **Канализация** — Sewage system
- **Интернет** — Internet availability

### 📑 Legal & Financial Options
- **Правоустанавливающие документы** — Ownership documents
- **Возможность обмена** — Exchange possibility
- **Возможность рассрочки** — Installment payment option
- **Возможность ипотеки** — Mortgage availability

### 📝 Additional Information
- **Разное** — Additional notes or features

---

## 🧹 Data Cleaning & Preprocessing
The following preprocessing steps were performed:
1. Handling missing values
2. Removing outliers and incorrect records
3. Encoding categorical features
4. Feature scaling for model compatibility

---

## 📈 Exploratory Data Analysis (EDA)
Exploratory data analysis was conducted using statistical and visualization techniques to:
- Examine price distributions
- Identify correlations between features and price
- Determine the most influential pricing factors

---

## 🤖 Modeling
Several regression models were trained and compared:
- Linear Regression (baseline)
- Random Forest Regressor
- XGBoost Regressor

---

## ✅ Model Evaluation
- Evaluation metric: **R² score**
- Best validation accuracy: **83%**
- XGBoost achieved the highest performance among all models

---

## 📊 Results & Insights
- Location and apartment size are the most significant price drivers
- Ensemble models outperform linear regression
- Non-linear models better capture complex relationships in housing data

## 🚀 How to Run the Project
1. Clone the repository:
```
git clone https://github.com/bbzet/predicting_the_cost_appartment
```
2. Install dependencies:
```
   pip install -r requirements.txt
```
3. Run the Jupyter notebooks directory sequentially.

Baiastan Zamirbekov
Data Science | Machine Learning | Python





