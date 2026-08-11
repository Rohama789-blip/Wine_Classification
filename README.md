# Wine Classification System

## Project Overview

Wine Classification System is a machine learning web application that predicts the class of a wine sample based on its chemical characteristics.

The application uses a trained Logistic Regression model and provides a simple web interface built with FastAPI and Jinja2.

## Features

* Wine classification using Machine Learning
* Logistic Regression model
* 13 wine characteristics as input
* Three wine classification classes
* Prediction probability and model confidence
* FastAPI REST API
* Jinja2-based web interface
* Responsive user interface
* Trained model saved using Joblib

## Technologies Used

* Python
* FastAPI
* Uvicorn
* Scikit-learn
* Pandas
* NumPy
* Joblib
* Jinja2
* HTML
* CSS

## Input Features

The model uses the following 13 features:

1. Alcohol
2. Malic Acid
3. Ash
4. Alcalinity of Ash
5. Magnesium
6. Total Phenols
7. Flavanoids
8. Nonflavanoid Phenols
9. Proanthocyanins
10. Color Intensity
11. Hue
12. OD280/OD315
13. Proline

## Machine Learning Model

The project uses Logistic Regression for wine classification.

The model was trained on the Wine dataset and evaluated using a train-test split.

### Model Performance

Test Accuracy:

**97.22%**

## Project Structure

```text
Wine_Classification/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│
├── models/
│   └── wine_model.pkl
│
├── static/
│   └── style.css
│
└── templates/
    └── index.html
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Rohama789-blip/Wine_Classification.git
```

Move into the project directory:

```bash
cd Wine_Classification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the FastAPI application using Uvicorn:

```bash
uvicorn app:app --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

## API Documentation

FastAPI automatically provides interactive API documentation.

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

## How It Works

1. The user enters the 13 chemical characteristics of a wine sample.
2. The FastAPI application receives the input.
3. The trained Logistic Regression model processes the values.
4. The model predicts the wine class.
5. The application displays the predicted class and confidence probability.

## Model Training

The model can be trained using:

```bash
python train_model.py
```

The trained model is saved in:

```text
models/wine_model.pkl
```

## Purpose

This project was developed as an educational machine learning and API deployment project to demonstrate an end-to-end workflow from model training to web-based prediction.

 
## License

This project is intended for educational and learning purposes.

```
```
