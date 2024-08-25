import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def manipulate_df(df):
    # Update sex column to numerical
    df['Sex'] = df['Sex'].map(lambda x: 0 if x == 'male' else 1)
    # Fill the NaN values in the age column
    df['Age'].fillna(value=df['Age'].mean(), inplace=True)
    # Create a first class column
    df['FirstClass'] = df['Pclass'].map(lambda x: 1 if x == 1 else 0)
    # Create a second class column
    df['SecondClass'] = df['Pclass'].map(lambda x: 1 if x == 2 else 0)
    # Create a third class column
    df['ThirdClass'] = df['Pclass'].map(lambda x: 1 if x == 3 else 0)
    return df

# Load and manipulate the data
train_df = pd.read_csv("train.csv")
manipulated_df = manipulate_df(train_df)

# Dropdown to select analytics
analytics_option = st.selectbox("Select Analytics to Display", 
                                ("Gender Distribution", 
                                 "Class Distribution", 
                                 "Age Distribution", 
                                 "Survived Distribution"))

# Show the selected analytics in the main layout
if analytics_option == "Gender Distribution":
    st.subheader("Gender Distribution")
    gender_counts = manipulated_df['Sex'].value_counts()
    gender_counts.index = ['Male', 'Female']  # Convert index to string labels
    st.bar_chart(gender_counts)

elif analytics_option == "Class Distribution":
    st.subheader("Class Distribution")
    class_counts = train_df['Pclass'].value_counts()
    class_counts.index = ['First Class', 'Second Class', 'Third Class']  # Convert index to string labels
    st.bar_chart(class_counts)

elif analytics_option == "Age Distribution":
    st.subheader("Age Distribution")
    age_groups = pd.cut(train_df['Age'], bins=[0, 12, 18, 30, 50, 80])
    age_group_counts = age_groups.value_counts().sort_index().rename(index=str)
    st.bar_chart(age_group_counts)

elif analytics_option == "Survived Distribution":
    st.subheader("Survived Distribution")
    survived_counts = train_df['Survived'].value_counts()
    survived_counts.index = ['Not Survived', 'Survived']  # Convert index to string labels
    st.bar_chart(survived_counts)

# Model training
features = manipulated_df[['Sex', 'Age', 'FirstClass', 'SecondClass', 'ThirdClass']]
survival = manipulated_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(features, survival, test_size=0.3, random_state=42)
scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)

model = LogisticRegression()
model.fit(train_features, y_train)

# Dropdown menu to select passenger
selected_name = st.selectbox("Select Passenger", train_df['Name'])

if selected_name:
    passenger_data = train_df[train_df['Name'] == selected_name]
    passenger_age = passenger_data['Age'].values[0]
    passenger_sex = passenger_data['Sex'].values[0]
    passenger_class = passenger_data['Pclass'].values[0]
    
    # Convert sex and class to numerical values
    sex = 0 if passenger_sex == 'male' else 1
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

    # Show prediction result
    if prediction[0] == 1:
        st.subheader(f'{selected_name} would have survived with a probability of {round(predict_probability[0][1] * 100, 2)}%.')
    else:
        st.subheader(f'{selected_name} would not have survived with a probability of {round(predict_probability[0][0] * 100, 2)}%.')
