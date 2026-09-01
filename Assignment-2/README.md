# Assignment 2: Machine Learning Training Pipeline

## Objective

The objective of this assignment is to build a complete machine learning workflow, including data reading, label creation, data splitting, exploratory data analysis, feature engineering, model training, hyperparameter tuning, and evaluation.

## Project Structure

- `01_read_and_join.ipynb`: Read the data and join the required tables.
- `02_create_labels.ipynb`: Create the target labels for the prediction task.
- `03_split_data.ipynb`: Split the data into training and testing sets.
- `04_eda_train_only.ipynb`: Perform exploratory data analysis using the training data.
- `05_feature_engineering.ipynb`: Prepare and transform features for machine learning.
- `06_train_tune_evaluate.ipynb`: Train, tune, and evaluate the machine learning model.
- `figures/`: Visualizations and figures generated during the analysis.
- `reports/`: Assignment reports and documentation.
- `your_database.db`: Local database used for the assignment.

## Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- SQLite

## How to Run

1. Clone the repository.
2. Install the required Python packages.
3. Open the notebooks in numerical order, from `01` to `06`.
4. Run each notebook cell by cell.

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook
