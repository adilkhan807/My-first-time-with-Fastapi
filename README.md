# Insurance Premium Predictor & Patient Management System

A Python-based project containing two backend applications:

1.  **Insurance Premium Category Predictor** --- a machine-learning
    application that predicts an insurance premium category from
    user-provided demographic, lifestyle, location, income, and
    occupation details.
2.  **Patient Management System** --- a REST API for creating, viewing,
    updating, deleting, and sorting patient records stored in a JSON
    file.

The project demonstrates machine learning model development, data
preprocessing, REST API development, input validation, and a Streamlit
user interface.

------------------------------------------------------------------------

## Table of Contents

-   [Project Overview](#project-overview)
-   [Features](#features)
-   [Technology Stack](#technology-stack)
-   [Project Structure](#project-structure)
-   [Application 1: Insurance Premium Category
    Predictor](#application-1-insurance-premium-category-predictor)
    -   [Dataset](#dataset)
    -   [Machine Learning Workflow](#machine-learning-workflow)
    -   [Input Features](#input-features)
    -   [Run the Prediction API](#run-the-prediction-api)
    -   [Run the Streamlit Frontend](#run-the-streamlit-frontend)
    -   [Prediction API Reference](#prediction-api-reference)
-   [Application 2: Patient Management
    System](#application-2-patient-management-system)
    -   [Run the Patient API](#run-the-patient-api)
    -   [Patient API Reference](#patient-api-reference)
-   [Installation](#installation)
-   [Important Configuration Notes](#important-configuration-notes)
-   [Future Improvements](#future-improvements)
-   [Disclaimer](#disclaimer)

------------------------------------------------------------------------

## Project Overview

### 1. Insurance Premium Category Predictor

This application uses a trained **Random Forest Classifier** to predict
an insurance premium category. The model uses engineered features
derived from the user's age, height, weight, smoking status, city,
income, and occupation.

The backend is built with FastAPI, while the user interface is built
with Streamlit.

### 2. Patient Management System

This application provides REST API endpoints to manage patient records.
It uses FastAPI and Pydantic for request validation and stores patient
information in a local JSON file.

It supports patient creation, retrieval, updating, deletion, and sorting
by selected attributes.

------------------------------------------------------------------------

## Features

### Insurance Premium Predictor

-   Predicts an insurance premium category using a trained machine
    learning pipeline.
-   Calculates BMI from height and weight.
-   Derives age groups and lifestyle risk categories.
-   Assigns a city tier based on predefined city lists.
-   Validates incoming data using Pydantic.
-   Provides a Streamlit interface for entering user details.
-   Exposes a prediction endpoint through FastAPI.

### Patient Management System

-   Create new patient records.
-   Retrieve all patient records.
-   Retrieve an individual patient by ID.
-   Update existing patient information.
-   Delete patient records.
-   Sort patient records by height, weight, or BMI.
-   Calculate BMI and a BMI-based verdict.
-   Validate request data using Pydantic.
-   Store records in a JSON file.

------------------------------------------------------------------------

## Technology Stack

  Technology                 Purpose
  -------------------------- -----------------------------------------------------
  Python                     Main programming language
  Pandas                     Data manipulation and feature preparation
  NumPy                      Numerical operations in model development
  Scikit-learn               Machine learning, preprocessing, and model pipeline
  Random Forest Classifier   Insurance category prediction
  FastAPI                    REST API development
  Pydantic                   Input validation and data models
  Streamlit                  Insurance predictor user interface
  Requests                   HTTP communication between frontend and API
  Pickle                     Saving and loading the trained model
  JSON                       Patient record storage
  Jupyter Notebook           Model development and experimentation

------------------------------------------------------------------------

## Project Structure

``` text
project/
│
├── app.py                  # FastAPI backend for insurance prediction
├── frontend.py             # Streamlit frontend for insurance prediction
├── FastAPI_ml_model.ipynb  # Machine learning model development notebook
├── insurance.csv           # Insurance dataset
├── model.pkl               # Saved machine learning pipeline
│
├── main (1).py             # FastAPI backend for patient management
├── patients.json           # Patient records in JSON format
│
└── README.md               # Project documentation
```

> **Note:** The patient API source file is currently named
> `main (1).py`. Rename it to `main.py` before running it with the
> Uvicorn command shown below.

------------------------------------------------------------------------

# Application 1: Insurance Premium Category Predictor

## Dataset

The model-development notebook uses `insurance.csv`.

The dataset contains **100 records** and the following eight columns:

  Column                         Description
  ------------------------------ ----------------------------------
  `age`                          Age of the individual
  `weight`                       Weight in kilograms
  `height`                       Height in meters
  `income_lpa`                   Annual income in lakhs per annum
  `smoker`                       Smoking status
  `city`                         City of residence
  `occupation`                   Occupation of the individual
  `insurance_premium_category`   Target category

The target variable is `insurance_premium_category`, with three
categories represented in the supplied dataset:

-   `Low`
-   `Medium`
-   `High`

## Machine Learning Workflow

The model is developed in `FastAPI_ml_model.ipynb` using the following
workflow:

1.  Load the dataset using Pandas.
2.  Explore the dataset and its columns.
3.  Create additional features.
4.  Separate input features and the target variable.
5.  Apply One-Hot Encoding to categorical features.
6.  Pass numerical features through the preprocessing pipeline.
7.  Split the data into training and testing sets.
8.  Train a Random Forest Classifier.
9.  Evaluate predictions using accuracy.
10. Save the complete preprocessing and model pipeline as `model.pkl`
    using Pickle.

### Feature Engineering

The notebook creates four additional features:

**1. Body Mass Index (BMI)**

``` python
bmi = weight / (height ** 2)
```

**2. Age Group**

  Age              Group
  ---------------- -------------
  Below 25         young
  25 to below 45   adult
  45 to below 60   middle_aged
  60 and above     senior

**3. Lifestyle Risk**

The application derives lifestyle risk using smoking status and BMI:

  Condition                 Risk
  ------------------------- --------
  Smoker and BMI above 30   high
  Smoker or BMI above 27    medium
  Otherwise                 low

**4. City Tier**

-   Tier 1: Cities in the predefined Tier 1 list.
-   Tier 2: Cities in the predefined Tier 2 list.
-   Tier 3: All other city names.

### Model Input Features

The trained pipeline uses these six features:

``` python
[
    "bmi",
    "age_group",
    "lifestyle_risk",
    "city_tier",
    "income_lpa",
    "occupation"
]
```

The target column is:

``` python
insurance_premium_category
```

## Run the Prediction API

The insurance prediction backend is implemented in `app.py`.

Make sure `app.py` and `model.pkl` are in the same directory.

Install the required packages:

``` bash
pip install fastapi uvicorn pandas scikit-learn
```

Start the API:

``` bash
uvicorn app:app --reload
```

The API will be available at:

``` text
http://127.0.0.1:8000
```

Open the interactive API documentation:

``` text
http://127.0.0.1:8000/docs
```

## Run the Streamlit Frontend

The frontend is implemented in `frontend.py`.

Install Streamlit and Requests:

``` bash
pip install streamlit requests
```

Run the frontend:

``` bash
streamlit run frontend.py
```

The frontend provides input fields for:

-   Age
-   Weight
-   Height
-   Annual income (LPA)
-   Smoking status
-   City
-   Occupation

Click **Predict Premium Category** to submit the details to the FastAPI
prediction endpoint.

> **Configuration:** The current `frontend.py` contains a hard-coded API
> URL. For local development, change it to
> `http://127.0.0.1:8000/predict` after starting the local API.

## Prediction API Reference

### `POST /predict`

Predicts the insurance premium category based on the submitted user
details.

**Example request:**

``` json
{
  "age": 30,
  "weight": 65,
  "height": 1.7,
  "income_lpa": 10,
  "smoker": false,
  "city": "Mumbai",
  "occupation": "private_job"
}
```

**Response format implemented in `app.py`:**

``` json
{
  "predicted_category": "Low"
}
```

The returned category is generated by the trained model. The example
category above is illustrative, not a guaranteed prediction.

### Input Validation

The API validates:

-   Age must be greater than 0 and less than 120.
-   Weight must be greater than 0.
-   Height must be greater than 0 and less than 2.5 meters.
-   Annual income must be greater than 0.
-   Smoking status must be a Boolean value.
-   Occupation must match one of the allowed values.

Allowed occupation values:

``` text
retired
freelancer
student
government_job
business_owner
unemployed
private_job
```

------------------------------------------------------------------------

# Application 2: Patient Management System

The Patient Management System is implemented using FastAPI and Pydantic.
Patient records are stored in `patients.json`.

## Run the Patient API

First, rename:

``` text
main (1).py
```

to:

``` text
main.py
```

Install the required packages:

``` bash
pip install fastapi uvicorn
```

Start the application:

``` bash
uvicorn main:app --reload
```

The API will be available at:

``` text
http://127.0.0.1:8000
```

Interactive API documentation:

``` text
http://127.0.0.1:8000/docs
```

Make sure `patients.json` is in the same working directory as `main.py`.

## Patient API Reference

  ----------------------------------------------------------------------------------
  Method                  Endpoint                           Description
  ----------------------- ---------------------------------- -----------------------
  GET                     `/`                                Returns the API welcome
                                                             message

  GET                     `/about`                           Returns information
                                                             about the API

  GET                     `/view`                            Retrieves all patient
                                                             records

  GET                     `/patient/{patient_id}`            Retrieves a patient by
                                                             ID

  GET                     `/sort?sort_by=height&order=asc`   Sorts patients by
                                                             height, weight, or BMI

  POST                    `/create`                          Creates a new patient

  PUT                     `/edit/{patient_id}`               Updates an existing
                                                             patient

  DELETE                  `/delete/{patient_id}`             Deletes a patient
  ----------------------------------------------------------------------------------

### 1. Get API Information

``` http
GET /
```

``` http
GET /about
```

### 2. View All Patients

``` http
GET /view
```

Returns all patient records from `patients.json`.

### 3. View a Specific Patient

``` http
GET /patient/P001
```

Replace `P001` with the required patient ID.

If the patient does not exist, the API returns a `404` error.

### 4. Sort Patients

``` http
GET /sort?sort_by=height&order=asc
```

Supported sorting fields:

-   `height`
-   `weight`
-   `bmi`

Supported sorting orders:

-   `asc` --- ascending order
-   `desc` --- descending order

Example:

``` http
GET /sort?sort_by=bmi&order=desc
```

### 5. Create a Patient

``` http
POST /create
```

**Example request:**

``` json
{
  "id": "P006",
  "name": "Rahul Sharma",
  "city": "Jaipur",
  "age": 29,
  "gender": "male",
  "height": 1.75,
  "weight": 70
}
```

The API calculates BMI and a BMI-based verdict before saving the patient
record.

### 6. Update a Patient

``` http
PUT /edit/P001
```

**Example request:**

``` json
{
  "city": "Delhi",
  "weight": 75
}
```

Only the supplied fields are updated.

### 7. Delete a Patient

``` http
DELETE /delete/P001
```

Replace `P001` with the ID of the patient to delete.

------------------------------------------------------------------------

## Installation

Follow these steps to set up the project locally.

### Step 1: Clone the Repository

``` bash
git clone <YOUR_REPOSITORY_URL>
```

Move into the project directory:

``` bash
cd <YOUR_PROJECT_DIRECTORY>
```

### Step 2: Create a Virtual Environment

``` bash
python -m venv venv
```

Activate it.

**Windows:**

``` bash
venv\Scripts\activate
```

**Linux / macOS:**

``` bash
source venv/bin/activate
```

### Step 3: Install Dependencies

For the insurance prediction application:

``` bash
pip install fastapi uvicorn pandas numpy scikit-learn streamlit requests
```

For the patient management application:

``` bash
pip install fastapi uvicorn
```

### Step 4: Start the Application You Need

**Insurance Prediction API:**

``` bash
uvicorn app:app --reload
```

**Insurance Streamlit Frontend:**

``` bash
streamlit run frontend.py
```

**Patient Management API:**

After renaming `main (1).py` to `main.py`:

``` bash
uvicorn main:app --reload
```

------------------------------------------------------------------------

## Important Configuration Notes

The following points are based on the supplied source files and should
be checked before deployment:

1.  **Frontend and backend response mismatch:** `app.py` returns
    `predicted_category` directly, while `frontend.py` currently checks
    for a `response` object containing `confidence` and
    `class_probabilities`. These response formats do not match. Update
    the frontend or backend so both use the same response structure.
2.  **API URL:** The Streamlit frontend currently points to a hard-coded
    IP address. Configure the correct API URL for your local or deployed
    environment.
3.  **Model compatibility:** `model.pkl` was created with scikit-learn
    1.6.1. Loading it under a different scikit-learn version may cause
    compatibility issues. Use a compatible environment or retrain and
    save the model with the installed version.
4.  **Patient BMI verdict:** The current `main (1).py` code labels BMI
    values from 25 to below 30 as `Normal`, while the supplied
    `patients.json` includes an `Overweight` verdict for one record.
    Review the verdict logic and existing records for consistency.
5.  **Patient data storage:** The patient API uses a JSON file rather
    than a database. This is suitable for a small demonstration but is
    not designed for concurrent production writes.
6.  **Training dataset size:** The supplied insurance dataset contains
    100 records. Model performance should be evaluated on more
    representative data before real-world use.

------------------------------------------------------------------------

## Future Improvements

Possible improvements include:

-   Add automated tests for API endpoints and model predictions.
-   Use environment variables for API configuration.
-   Add proper error handling and response validation in the Streamlit
    frontend.
-   Add confidence scores and class probabilities if the backend is
    extended to return them.
-   Improve model evaluation with additional metrics and
    cross-validation.
-   Increase the size and diversity of the training dataset.
-   Replace JSON storage with a relational database such as PostgreSQL
    or MySQL.
-   Add authentication and authorization to patient management
    endpoints.
-   Add logging and deployment configuration.

------------------------------------------------------------------------

## Disclaimer

This project is intended for educational and demonstration purposes. The
insurance category predictions are machine-learning outputs based on the
supplied dataset and should not be treated as actual insurance quotes,
financial advice, or underwriting decisions.

The Patient Management System uses sample patient records. Do not use
real or sensitive patient information in an unsecured development
environment.

------------------------------------------------------------------------

## Author

**Your Name**

If you find this project useful, consider giving the repository a star
on GitHub.
