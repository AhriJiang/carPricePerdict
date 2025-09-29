# Used Car Price Prediction Project

## Project Introduction

This project aims to predict used car prices using machine learning techniques, particularly the CatBoost algorithm. It includes a complete data processing pipeline, feature engineering methods, model training and evaluation functions, as well as an independent prediction script.

## Key Features

- **Data Preprocessing**: Handling missing values, outliers, and date format conversion
- **Feature Engineering**:
  - Creating time-related features (vehicle age, registration season, etc.)
  - Generating vehicle-related features
  - Building statistical features (brand-level price statistics, etc.)
  - Implementing cross features (brand+model combinations, etc.)
- **Model Training**: Training regression models using the CatBoost algorithm
- **Model Evaluation**: Calculating RMSE, MAE, R2 and other evaluation metrics
- **Feature Importance Analysis**: Visualizing key predictive features
- **Independent Prediction**: Supporting prediction on new data using saved models

## Project Structure

```
e:\TianChi\carPrice\
├── feature_engineering_and_catboost.py    # Main script: feature engineering and model training
├── feature_engineering_and_catboost.ipynb # Jupyter Notebook version
├── predict_with_saved_model.py           # Independent prediction script
├── processed_data/                       # Processed data and saved models
│   ├── fe_X_train.joblib                 # Training feature data
│   ├── fe_X_val.joblib                   # Validation feature data
│   ├── fe_y_train.joblib                 # Training target data
│   ├── fe_y_val.joblib                   # Validation target data
│   ├── fe_test_data.joblib               # Test feature data
│   ├── fe_sale_ids.joblib                # Test ID data
│   ├── fe_cat_features.joblib            # Categorical feature list
│   └── fe_catboost_model.cbm             # Saved CatBoost model
├── used_car_train_20200313.csv           # Training dataset
├── used_car_testB_20200421.csv           # Test dataset
└── predict_test_result.csv               # Prediction result output
```

## Technology Stack

- **Python 3.7+**
- **Pandas**: Data processing and analysis
- **NumPy**: Scientific computing
- **Matplotlib/Seaborn**: Data visualization
- **CatBoost**: Gradient boosting decision tree algorithm
- **Scikit-learn**: Model evaluation and data splitting
- **Joblib**: Model and data persistence

## Installation Instructions

### 1. Clone or Download the Project

```bash
git clone <repository_url>
cd e:\TianChi\carPrice
```

### 2. Install Dependencies

Install the required Python packages using pip:

```bash
pip install pandas numpy matplotlib seaborn catboost scikit-learn joblib
```

### 3. Prepare Data

Ensure the following data files exist in the project root directory:
- `used_car_train_20200313.csv`: Training dataset
- `used_car_testB_20200421.csv`: Test dataset

## Usage

### Method 1: Run Complete Feature Engineering and Model Training

```bash
python feature_engineering_and_catboost.py
```

This script will:
- Load and preprocess data
- Perform feature engineering
- Train and evaluate the model
- Generate prediction results
- Save the model and processed data

### Method 2: Use Saved Model for Prediction

If you have already trained a model, you can directly use the independent prediction script:

```bash
python predict_with_saved_model.py
```

This script will:
- Load the saved model
- Load the processed test data
- Make predictions and save results

## Model Performance

The trained CatBoost model's performance on the validation set:

- **Root Mean Squared Error (RMSE)**: 1299.96
- **Mean Absolute Error (MAE)**: 535.64
- **R2 Score**: 0.9688

## Important Features

The top 10 important features of the model (sorted by importance):

1. v_3 (25.88%)
2. v_0 (20.22%)
3. v_12 (15.68%)
4. v_8 (3.88%)
5. v_9 (3.20%)
6. v_6 (3.05%)
7. v_10 (2.28%)
8. kilometer_brand_ratio (2.05%)
9. power (1.97%)
10. v_14 (1.84%)

## Contribution Guidelines

Contributions and questions are welcome! Please follow these steps:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Contact

For questions or suggestions, please contact:

- Project Repository: <repository_url>

---

*Last Updated: 2024*