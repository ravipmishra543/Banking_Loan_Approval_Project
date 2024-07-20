import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import logging
import joblib
from sklearn.model_selection import train_test_split
from src.data_loading import load_data
from src.preprocessing import encode_labels, drop_unnecessary_columns, define_features_and_target
from src.visualization import plot_heatmap, plot_count
from src.model import train_logistic_regression, train_random_forest, train_decision_tree, train_knn
from src.evaluation import evaluate_model

# Configure logging
logging.basicConfig(filename='loan_approval_project.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    logging.info('Started the main function')
    
    data_path = 'data/dataset.CSV'
    logging.info(f'Loading data from {data_path}')
    df = load_data(data_path)

    logging.info('Data loaded successfully')
    logging.info(f'DataFrame columns: {df.columns}')

    # Plotting correlation heatmap
    logging.info('Plotting correlation heatmap')
    plot_heatmap(df, save_path='plots/heatmap.png')

    # Count plots
    logging.info('Plotting count plots for various columns')
    plot_count(df, "NEW_CUST", hue='STATUS', save_path='plots/count_plot_new_cust.png')
    plot_count(df, "P_RESTYPE", hue='STATUS', save_path='plots/count_plot_p_restype.png')
    plot_count(df, "P_CATEGORY", hue='STATUS', save_path='plots/count_plot_p_category.png')
    plot_count(df, "STATUS", save_path='plots/count_plot_status.png')
    plot_count(df, "AGE", save_path='plots/count_plot_age.png')
    plot_count(df, "SEX", save_path='plots/count_plot_sex.png')
    plot_count(df, "NO_OF_DEPENDENTS", save_path='plots/count_plot_no_of_dependents.png')
    plot_count(df, "MARITAL", save_path='plots/count_plot_marital.png')
    plot_count(df, "INCOM_EXP_GMI", save_path='plots/count_plot_incom_exp_gmi.png')

    # Preprocessing
    logging.info('Encoding labels')
    df = encode_labels(df)
    logging.info('Dropping unnecessary columns')
    df = drop_unnecessary_columns(df)
    
    # Use the actual target column name
    target_column = 'STATUS'
    logging.info(f'Defining features and target with target column: {target_column}')
    X, y = define_features_and_target(df, target_column=target_column)

    # Splitting the data
    logging.info('Splitting the data into training and test sets')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training models
    logging.info('Training Logistic Regression model')
    logreg_cv = train_logistic_regression(X_train, y_train)
    joblib.dump(logreg_cv.best_estimator_, 'models/best_logistic_regression_model.pkl')
    logging.info('Best Logistic Regression model saved')

    logging.info('Training Random Forest model')
    rf_tuning = train_random_forest(X_train, y_train)
    joblib.dump(rf_tuning.best_estimator_, 'models/best_random_forest_model.pkl')
    logging.info('Best Random Forest model saved')

    logging.info('Training Decision Tree model')
    dtc_gs = train_decision_tree(X_train, y_train)
    joblib.dump(dtc_gs.best_estimator_, 'models/best_decision_tree_model.pkl')
    logging.info('Best Decision Tree model saved')

    logging.info('Training KNN model')
    knn_gs = train_knn(X_train, y_train)
    joblib.dump(knn_gs.best_estimator_, 'models/best_knn_model.pkl')
    logging.info('Best KNN model saved')

    # Evaluating models
    logging.info('Evaluating Logistic Regression model')
    logreg_accuracy, logreg_report = evaluate_model(logreg_cv, X_test, y_test)
    logging.info(f'Logistic Regression Accuracy: {logreg_accuracy}')
    logging.info(f'Logistic Regression Report:\n{logreg_report}')
    with open('results/logistic_regression_report.txt', 'w') as f:
        f.write(f'Logistic Regression Accuracy: {logreg_accuracy}\n')
        f.write(f'{logreg_report}\n')

    logging.info('Evaluating Random Forest model')
    rf_accuracy, rf_report = evaluate_model(rf_tuning, X_test, y_test)
    logging.info(f'Random Forest Accuracy: {rf_accuracy}')
    logging.info(f'Random Forest Report:\n{rf_report}')
    with open('results/random_forest_report.txt', 'w') as f:
        f.write(f'Random Forest Accuracy: {rf_accuracy}\n')
        f.write(f'{rf_report}\n')

    logging.info('Evaluating Decision Tree model')
    dtc_accuracy, dtc_report = evaluate_model(dtc_gs, X_test, y_test)
    logging.info(f'Decision Tree Accuracy: {dtc_accuracy}')
    logging.info(f'Decision Tree Report:\n{dtc_report}')
    with open('results/decision_tree_report.txt', 'w') as f:
        f.write(f'Decision Tree Accuracy: {dtc_accuracy}\n')
        f.write(f'{dtc_report}\n')

    logging.info('Evaluating KNN model')
    knn_accuracy, knn_report = evaluate_model(knn_gs, X_test, y_test)
    logging.info(f'KNN Accuracy: {knn_accuracy}')
    logging.info(f'KNN Report:\n{knn_report}')
    with open('results/knn_report.txt', 'w') as f:
        f.write(f'KNN Accuracy: {knn_accuracy}\n')
        f.write(f'{knn_report}\n')

    logging.info('Main function execution completed')

if __name__ == '__main__':
    main()