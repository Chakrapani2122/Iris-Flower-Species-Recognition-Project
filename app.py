import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Function to train and save the model if not already fitted
def train_and_save_model():
    # Load the dataset
    df = pd.read_csv("Iris.csv")
    df = df.drop(columns="Id")  # Drop unnecessary column
    X = df.drop("Species", axis=1)
    y = df["Species"]

    # Split the data
    X_train, _, y_train, _ = train_test_split(X, y, random_state=0)

    # Train the model
    model = SVC(C=0.1, gamma=0.1, kernel='poly')
    model.fit(X_train, y_train)

    # Save the model
    with open("best_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    return model

# Load the trained model or train it if not fitted
try:
    with open("best_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    if not hasattr(model, "support_"):
        st.warning("The loaded model is not fitted. Training a new model...")
        model = train_and_save_model()
except (FileNotFoundError, EOFError, pickle.UnpicklingError):
    st.warning("Model file not found or corrupted. Training a new model...")
    model = train_and_save_model()

# Streamlit app title
st.title("Iris Flower Species Recognition")

# Input fields for flower measurements
st.header("Enter the measurements:")
col1, col2 = st.columns(2)  # Create 2 columns

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
with col2:
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Predict button
if st.button("Classify"):
    # Ensure all inputs are provided
    if sepal_length > 0 and sepal_width > 0 and petal_length > 0 and petal_width > 0:
        # Prepare input for the model
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_features)
        
        # Display the result
        st.success(f"The predicted Iris species is: {prediction[0]}")

        # Display images and facts based on the prediction
        if prediction[0] == "Iris-setosa":
            col1, col2 = st.columns(2)  # Create two columns
            with col1:
                img = mpimg.imread("assets/setosa_1.jpeg")
                plt.imshow(img)
                plt.axis('off')
                st.pyplot(plt)
            with col2:
                img = mpimg.imread("assets/setosa_2.jpeg")
                plt.imshow(img)
                plt.axis('off')
                st.pyplot(plt)
            st.markdown("""
            ### 🌼 Meet the Competitors!
            **Iris Setosa – The Petite Powerhouse 💜**
            
            - Sepal Length: 4.3 - 5.8 cm
            - Sepal Width: 2.3 - 4.4 cm
            - Petal Length: 1.0 - 1.9 cm
            - Petal Width: 0.1 - 0.6 cm
            - Specialty: Smallest petals and round-tipped petals.
            """)
        elif prediction[0] == "Iris-versicolor":
            col1, col2 = st.columns(2)  # Create two columns
            with col1:
                img = mpimg.imread("assets/versicolor_1.jpeg")
                plt.imshow(img)
                plt.axis('off')
                st.pyplot(plt)
            with col2:
                img = mpimg.imread("assets/versicolor_2.jpeg")
                plt.imshow(img)
                plt.axis('off')
                st.pyplot(plt)
            st.markdown("""
            ### 🌼 Meet the Competitors!
            **Iris Versicolor – The Balanced Beauty 💙**
            
            - Sepal Length: 4.9 - 7.0 cm
            - Sepal Width: 2.0 - 3.4 cm
            - Petal Length: 3.0 - 5.1 cm
            - Petal Width: 1.0 - 1.8 cm
            - Specialty: Balanced proportions and slender petals.
            """)
        elif prediction[0] == "Iris-virginica":
            col1, col2 = st.columns(2)  # Create two columns
            with col1:
                img = mpimg.imread("assets/virginica_1.jpeg")
                plt.imshow(img)
                plt.axis('off')
                st.pyplot(plt)
            with col2:
                img = mpimg.imread("assets/virginica_2.jpeg")
                plt.imshow(img)
                plt.axis('off')
                st.pyplot(plt)
            st.markdown("""
            ### 🌼 Meet the Competitors!
            **Iris Virginica – The Tall Queen 👑**
            
            - Sepal Length: 4.9 - 7.9 cm
            - Sepal Width: 2.2 - 3.8 cm
            - Petal Length: 4.5 - 6.9 cm
            - Petal Width: 1.4 - 2.5 cm
            - Specialty: Largest petals and thrives in wetlands.
            """)
    else:
        st.error("Please provide all measurements to classify the flower.")
