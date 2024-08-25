import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Function to manipulate and prepare the data
def manipulate_df(df):
    df['Sex'] = df['Sex'].map(lambda x: 0 if x == 'male' else 1)
    df['Age'].fillna(value=df['Age'].mean(), inplace=True)
    df['FirstClass'] = df['Pclass'].map(lambda x: 1 if x == 1 else 0)
    df['SecondClass'] = df['Pclass'].map(lambda x: 1 if x == 2 else 0)
    df['ThirdClass'] = df['Pclass'].map(lambda x: 1 if x == 3 else 0)
    return df

# Load and manipulate the data
train_df = pd.read_csv("train.csv")
manipulated_df = manipulate_df(train_df)

# Model training
features = manipulated_df[['Sex', 'Age', 'FirstClass', 'SecondClass', 'ThirdClass']]
survival = manipulated_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(features, survival, test_size=0.3, random_state=42)
scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)

model = LogisticRegression()
model.fit(train_features, y_train)

# Make predictions
y_predict = model.predict(test_features)
train_score = model.score(train_features, y_train)
test_score = model.score(test_features, y_test)

# Confusion matrix
confusion = confusion_matrix(y_test, y_predict)
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]

# App title
st.title("Titanic Survival Prediction App")

st.write("""
    Welcome to the Titanic Survival Prediction App!

    This application allows you to analyze and predict the survival of passengers aboard the Titanic based on their attributes. 
    You can explore various analytics such as gender distribution, class distribution, and age distribution of passengers. 

    Additionally, you can select a specific passenger to see detailed information including their predicted survival probability and 
    whether they would have survived the disaster according to our logistic regression model.

    Use the sidebar to choose different analytics or select a passenger to see their details and prediction.
""")


# Sidebar for analytics and passenger selection
st.sidebar.header("Passenger Analytics")
selected_analytics = st.sidebar.selectbox("Select Analytics", ["None", "Gender Distribution", "Class Distribution", "Age Distribution"])

# Sidebar for selecting passenger
selected_name = st.sidebar.selectbox("Select Passenger", ["None"] + list(train_df['Name'].unique()))

if selected_name != "None":
    passenger_data = train_df[train_df['Name'] == selected_name]
    passenger_age = passenger_data['Age'].values[0]
    passenger_sex = passenger_data['Sex'].values[0]
    passenger_class = passenger_data['Pclass'].values[0]
    passenger_survived = passenger_data['Survived'].values[0]
    
    # Convert to numerical values for predictions
    sex = passenger_sex
    f_class, s_class, t_class = 0, 0, 0
    if passenger_class == 1:
        f_class = 1
    elif passenger_class == 2:
        s_class = 1
    else:
        t_class = 1
    
    # Predict survival
    input_data = scaler.transform([[sex, passenger_age, f_class, s_class, t_class]])
    prediction = model.predict(input_data)
    predict_probability = model.predict_proba(input_data)

    survival_probability = round(predict_probability[0][1] * 100, 2)
    not_survival_probability = round(predict_probability[0][0] * 100, 2)

    # Show passenger details
    st.subheader(f"Details for {selected_name}")
    st.write(f"Gender: {'Male' if passenger_sex == 0 else 'Female'}")
    st.write(f"Age: {passenger_age}")
    st.write(f"Class: {'First Class' if passenger_class == 1 else 'Second Class' if passenger_class == 2 else 'Third Class'}")
    st.write(f"Outcome: {'Survived' if passenger_survived == 1 else 'Not Survived'}")

    # Show prediction result
    if prediction[0] == 1:
        st.subheader(f'{selected_name} survived with a probability of {survival_probability}%.')
    else:
        st.subheader(f'{selected_name} would not have survived with a probability of {survival_probability}%.')
else:
    # Only show analytics and confusion matrix if no passenger is selected
    if selected_analytics == "Gender Distribution":
        st.subheader("Gender Distribution")
        gender_counts = manipulated_df['Sex'].map({0: 'Male', 1: 'Female'}).value_counts()
        st.bar_chart(gender_counts)

    elif selected_analytics == "Class Distribution":
        st.subheader("Class Distribution")
        class_counts = train_df['Pclass'].map({1: 'First Class', 2: 'Second Class', 3: 'Third Class'}).value_counts()
        st.bar_chart(class_counts)

    elif selected_analytics == "Age Distribution":
        st.subheader("Age Distribution")
        age_groups = pd.cut(train_df['Age'], bins=[0, 12, 18, 30, 50, 80])
        age_group_counts = age_groups.value_counts().sort_index().rename(index=str)
        st.bar_chart(age_group_counts)

    # Display confusion matrix and metrics
    st.subheader("Train Set Score: {}".format(round(train_score, 3)))
    st.subheader("Test Set Score: {}".format(round(test_score, 3)))

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(['False Negative', 'True Negative', 'True Positive', 'False Positive'], [FN, TN, TP, FP])
    ax.set_xlabel('Confusion Matrix')
    st.pyplot(fig)
