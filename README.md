# Titanic Survival Prediction

This is a simple Streamlit app template for predicting the survival of passengers aboard the Titanic based on their attributes. You can modify and customize this app to suit your needs.

## How to Run the App

To run the app on your own machine, follow these steps:

1. Install the required dependencies by running the following command in your terminal:

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app using the following command:

   ```
   $ streamlit run streamlit_app.py
   ```

## Functionality

The app provides the following functionality:

### Data Manipulation

The `manipulate_df` function is used to manipulate and prepare the data. It performs the following transformations on the input DataFrame:

- Maps the 'Sex' column to numerical values (0 for 'male' and 1 for 'female').
- Fills missing values in the 'Age' column with the mean age.
- Creates new binary columns ('FirstClass', 'SecondClass', 'ThirdClass') based on the 'Pclass' column.

### Model Training

The app uses logistic regression for survival prediction. The model is trained using the following steps:

- The data is split into training and testing sets using the `train_test_split` function.
- The features are standardized using the `StandardScaler`.
- The logistic regression model is instantiated and trained using the training features and labels.

### Prediction

The app allows you to select a specific passenger and see their predicted survival probability and outcome according to the trained logistic regression model. The following steps are performed:

- The selected passenger's attributes are converted to numerical values.
- The input data is standardized using the same scaler used during training.
- The model predicts the survival outcome and probability for the selected passenger.

### Analytics

The app provides various analytics options that you can select from the sidebar:

- Gender Distribution: Shows the distribution of male and female passengers.
- Class Distribution: Shows the distribution of passengers across different classes.
- Age Distribution: Shows the distribution of passenger ages.

### Confusion Matrix

The app displays a confusion matrix and metrics (train set score and test set score) to evaluate the performance of the logistic regression model.

## Credits

This app template was developed by [Zain Haidar](https://zaintheanalyst.com). For more information, please visit the [GitHub repository](https://github.com/zainhaidar16).

