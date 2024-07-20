# Banking Loan Approval Project

This project aims to build and evaluate several machine learning models to predict loan approval status based on customer data. The project includes data preprocessing, model training, evaluation, and comparison. The best-performing model is saved for future use.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Banking_Loan_Approval_Project.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Banking_Loan_Approval_Project
    ```

3. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place the dataset file (`dataset.CSV`) in the `data/` directory.

2. Run the main script:
    ```bash
    python main.py
    ```

3. The script will perform the following tasks:
    - Load and preprocess the data.
    - Generate and save plots in the `plots/` directory.
    - Train and evaluate multiple machine learning models.
    - Save detailed evaluation reports in the `results/` directory.
    - Compare the models and save the best-performing model in the `models` directory.
    - Log all activities in `loan_approval_project.log`.

## Dataset

The dataset used in this project contains information about loan applications and their approval status. Below are the details of the columns in the dataset:

- `APP_ID`: Unique identifier for each application.
- `CIBIL_SCORE_VALUE`: Credit score of the applicant.
- `NEW_CUST`: Indicates if the applicant is a new customer (Yes/No).
- `CUS_CATGCODE`: Customer category code.
- `EMPLOYMENT_TYPE`: Type of employment (e.g., Salaried, Self-Employed).
- `AGE`: Age of the applicant.
- `SEX`: Gender of the applicant.
- `NO_OF_DEPENDENTS`: Number of dependents.
- `MARITAL`: Marital status of the applicant.
- `EDU_QUA`: Educational qualification.
- `P_RESTYPE`: Type of residence.
- `P_CATEGORY`: Residence category.
- `EMPLOYEE_TYPE`: Type of employee.
- `MON_IN_OCC`: Monthly income occupation.
- `INCOM_EXP_GMI`: Income expense gross monthly income.
- `LTV`: Loan to value ratio.
- `TENURE`: Tenure of the loan.
- `STATUS`: Loan approval status (approved/denied).

## Models

The project includes the following machine learning models:
- Logistic Regression
- Random Forest
- Decision Tree
- K-Nearest Neighbors (KNN)

## Evaluation

Models are evaluated using the following metrics:
- Accuracy
- Classification Report (precision, recall, f1-score)

Evaluation reports are saved in the `results/` directory.

## Results

The best-performing model is saved in the `models` directory as `best_model.pkl`.

## Logging

All steps and activities are logged in the `loan_approval_project.log` file. This includes data loading, preprocessing, model training, evaluation, and saving the best model.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.