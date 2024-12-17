import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the dataset
def load_data():
    data = pd.read_csv('ProcessedCopperSet.csv')  # Path to your cleaned CSV file
    label_encoder = LabelEncoder()
    data['status'] = label_encoder.fit_transform(data['status'])  # Convert WON/LOST to 1/0
    data['item_type'] = label_encoder.fit_transform(data['item_type'])  # Convert categorical to numerical
    data['application'] = label_encoder.fit_transform(data['application'])  # Convert categorical to numerical
    data['material_ref'] = label_encoder.fit_transform(data['material_ref'])  # Convert categorical to numerical
    data['product_ref'] = label_encoder.fit_transform(data['product_ref'])  # Convert categorical to numerical
    return data, label_encoder

# Split data into features and target
def prepare_data(data):
    X = data[['quantity_tons', 'item_type', 'application', 'thickness', 'width', 'material_ref', 'product_ref']]
    y = data['status']
    return X, y

# Train the models
def train_models(X_train, y_train):
    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)

    # Gradient Boosting Classifier
    gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_classifier.fit(X_train, y_train)

    return rf_classifier, dt_classifier, gb_classifier

# Main Streamlit app
def main():
    st.title("Copper Industry Lead Prediction")
    st.write("This app allows you to predict the status of copper industry leads using multiple machine learning models.")

    # Load data
    data, label_encoder = load_data()

    # Prepare features and target
    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    rf_classifier, dt_classifier, gb_classifier = train_models(X_train, y_train)

    # Sidebar for user input
    st.sidebar.title("Input Features")
    inputs = {}
    for col in X.columns:
        inputs[col] = st.sidebar.number_input(f"Enter value for {col}", value=0.0)

    # Choose model
    model_choice = st.sidebar.selectbox("Choose a model for prediction:", ["Random Forest", "Decision Tree", "Gradient Boosting"])

    if st.sidebar.button("Predict Status"):
        input_data = pd.DataFrame([inputs])
        if model_choice == "Random Forest":
            prediction = rf_classifier.predict(input_data)
            status = label_encoder.inverse_transform(prediction)[0]
            st.success(f"The predicted status using Random Forest is: {status}")
        elif model_choice == "Decision Tree":
            prediction = dt_classifier.predict(input_data)
            status = label_encoder.inverse_transform(prediction)[0]
            st.success(f"The predicted status using Decision Tree is: {status}")
        elif model_choice == "Gradient Boosting":
            prediction = gb_classifier.predict(input_data)
            status = label_encoder.inverse_transform(prediction)[0]
            st.success(f"The predicted status using Gradient Boosting is: {status}")

    # Show model performance metrics
    st.sidebar.title("Model Performance Metrics")
    if st.sidebar.button("Evaluate Models"):
        # Evaluate Random Forest
        rf_y_pred = rf_classifier.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_y_pred)
        st.subheader("Random Forest Classifier Performance")
        st.write(f"Accuracy: {rf_accuracy:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, rf_y_pred, zero_division=0))

        # Evaluate Decision Tree
        dt_y_pred = dt_classifier.predict(X_test)
        dt_accuracy = accuracy_score(y_test, dt_y_pred)
        st.subheader("Decision Tree Classifier Performance")
        st.write(f"Accuracy: {dt_accuracy:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, dt_y_pred, zero_division=0))

        # Evaluate Gradient Boosting
        gb_y_pred = gb_classifier.predict(X_test)
        gb_accuracy = accuracy_score(y_test, gb_y_pred)
        st.subheader("Gradient Boosting Classifier Performance")
        st.write(f"Accuracy: {gb_accuracy:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, gb_y_pred, zero_division=0))

if __name__ == "__main__":
    main()
