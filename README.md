# 🌸 Iris Flower Species Recognition Project

A machine learning-based web application that classifies Iris flower species based on sepal and petal measurements. This project uses a Support Vector Machine (SVM) model trained on the famous Iris dataset and provides an interactive Streamlit web interface for real-time predictions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Iris Species](#iris-species)
- [Requirements](#requirements)
- [Running the Application](#running-the-application)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The **Iris Flower Species Recognition Project** is a comprehensive machine learning system designed to automatically classify Iris flowers into three distinct species: *Iris-setosa*, *Iris-versicolor*, and *Iris-virginica*. The project demonstrates the complete machine learning workflow including:

- **Data Exploration & Preprocessing**: Loading and cleaning the Iris dataset
- **Model Development**: Training multiple machine learning algorithms (Logistic Regression, Decision Trees, Random Forest, SVM)
- **Hyperparameter Tuning**: Using GridSearchCV for optimal model configuration
- **Model Evaluation**: Assessing accuracy and generating classification reports
- **Interactive Web Application**: Providing a user-friendly Streamlit interface for predictions

The final trained model leverages Support Vector Machine (SVM) technology with optimized hyperparameters to deliver accurate species classification based on four key floral measurements: sepal length, sepal width, petal length, and petal width.

---

## ✨ Features

### Core Features
- **Accurate Classification**: Uses SVM algorithm with optimized parameters (C=0.1, gamma=0.1, kernel='poly')
- **Real-time Predictions**: Instant classification results through an intuitive web interface
- **Visual Feedback**: Displays species-specific images and characteristics after prediction
- **Error Handling**: Robust exception management for model loading and data validation
- **Model Persistence**: Pre-trained model saved as `best_model.pkl` for quick inference

### Web Application Features
- **User-Friendly Interface**: Built with Streamlit for ease of use
- **Input Validation**: Ensures all measurements are provided before classification
- **Interactive Layout**: Two-column input design for better UX
- **Species Information**: Detailed characteristics and measurement ranges for each species
- **Visual Gallery**: Species-specific images (2 per species) for reference

---

## 📊 Dataset

### Dataset Overview
The project uses the **Iris Dataset**, one of the most famous datasets in machine learning and statistics. This dataset was introduced by Ronald Fisher and contains measurements of iris flowers.

### Dataset Characteristics
- **Total Records**: 150 samples (50 per species)
- **Features**: 4 numerical attributes
  - **Sepal Length** (cm): Length of the outer covering of the flower
  - **Sepal Width** (cm): Width of the outer covering
  - **Petal Length** (cm): Length of the flower's petals
  - **Petal Width** (cm): Width of the flower's petals
- **Target Classes**: 3 iris species
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica
- **Data Quality**: No missing values; clean and well-structured
- **File**: `Iris.csv` included in the repository

### Data Preprocessing Steps
1. Load dataset from CSV file
2. Check for missing values (none found)
3. Drop the 'Id' column (non-informative)
4. Separate features (X) from target variable (y)
5. Split data into training and testing sets (75-25 split by default)

---

## 📁 Project Structure

```
Iris-Flower-Species-Recognition-Project/
│
├── README.md                    # Project documentation (this file)
├── app.py                       # Streamlit web application
├── Project.ipynb                # Jupyter notebook with full analysis
├── Iris.csv                     # Dataset file (150 samples)
├── best_model.pkl               # Pre-trained SVM model
├── requirements.txt             # Python dependencies
│
└── assets/                      # Image assets for web app
    ├── setosa_1.jpeg           # Setosa species images
    ├── setosa_2.jpeg
    ├── versicolor_1.jpeg       # Versicolor species images
    ├── versicolor_2.jpeg
    ├── virginica_1.jpeg        # Virginica species images
    └── virginica_2.jpeg
```

---

## 🔧 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Chakrapani2122/Iris-Flower-Species-Recognition-Project.git
   cd Iris-Flower-Species-Recognition-Project
   ```

2. **Create a Virtual Environment** (recommended)
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Running the Web Application

1. **Start the Streamlit Application**
   ```bash
   streamlit run app.py
   ```

2. **Access the Application**
   - The app will automatically open in your default browser
   - URL: `http://localhost:8501` (or as shown in the terminal)

3. **Using the Application**
   - Enter the four measurements (Sepal Length, Sepal Width, Petal Length, Petal Width)
   - Click the "Classify" button
   - View the predicted species and its characteristics
   - See reference images of the identified species

### Example Usage

**Input Values:**
- Sepal Length: 5.1 cm
- Sepal Width: 3.5 cm
- Petal Length: 1.4 cm
- Petal Width: 0.2 cm

**Expected Output:** Iris-setosa with relevant images and characteristics

---

## 🤖 Model Details

### Model Architecture
- **Algorithm**: Support Vector Machine (SVM)
- **Type**: Classification (Multi-class)
- **Framework**: scikit-learn

### Hyperparameters
```python
Model Configuration:
- C = 0.1                  # Regularization parameter
- gamma = 0.1              # Kernel coefficient
- kernel = 'poly'          # Polynomial kernel function
- random_state = 0         # For reproducibility
```

### Training Process
1. **Data Split**: 75% training, 25% testing
2. **Feature Scaling**: Not explicitly applied (SVM with these params performs well)
3. **Model Training**: Fitted on training data
4. **Evaluation**: Accuracy and classification metrics computed on test set
5. **Hyperparameter Optimization**: GridSearchCV used to find optimal parameters
6. **Model Persistence**: Final model saved using Pickle for inference

### Model Performance
The SVM model with polynomial kernel and optimized hyperparameters achieves high accuracy on the Iris dataset, typically exceeding 95% on test data.

### Algorithms Explored
During development, multiple algorithms were evaluated:
- **Logistic Regression**: Linear classification approach
- **Decision Tree**: Tree-based classification
- **Random Forest**: Ensemble method with multiple decision trees
- **SVM**: Final selected model with best performance

---

## 🌸 Iris Species

### 1. Iris-Setosa 💜 (The Petite Powerhouse)

**Characteristics:**
- **Sepal Length**: 4.3 - 5.8 cm
- **Sepal Width**: 2.3 - 4.4 cm
- **Petal Length**: 1.0 - 1.9 cm
- **Petal Width**: 0.1 - 0.6 cm
- **Key Features**: 
  - Smallest petals among the three species
  - Round-tipped petals
  - Compact and sturdy appearance
  - Native to rocky habitats

---

### 2. Iris-Versicolor 💙 (The Balanced Beauty)

**Characteristics:**
- **Sepal Length**: 4.9 - 7.0 cm
- **Sepal Width**: 2.0 - 3.4 cm
- **Petal Length**: 3.0 - 5.1 cm
- **Petal Width**: 1.0 - 1.8 cm
- **Key Features**:
  - Balanced proportions between species
  - Slender and pointed petals
  - Medium-sized flowers
  - Highly variable characteristics

---

### 3. Iris-Virginica 👑 (The Tall Queen)

**Characteristics:**
- **Sepal Length**: 4.9 - 7.9 cm
- **Sepal Width**: 2.2 - 3.8 cm
- **Petal Length**: 4.5 - 6.9 cm
- **Petal Width**: 1.4 - 2.5 cm
- **Key Features**:
  - Largest petals among the three species
  - Long and elegant petals
  - Thrives in wetland environments
  - Regal and striking appearance

---

## 📦 Requirements

### Python Dependencies

The project requires the following Python packages (see `requirements.txt`):

```
streamlit==1.24.0          # Web application framework
numpy==1.24.3              # Numerical computing library
pandas==1.5.3              # Data manipulation and analysis
scikit-learn==1.2.2        # Machine learning library
```

### System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python Version**: 3.7 or higher
- **RAM**: Minimum 2GB (4GB recommended)
- **Disk Space**: ~100MB for dependencies and project files

---

## ▶️ Running the Application

### Quick Start

```bash
# 1. Navigate to project directory
cd Iris-Flower-Species-Recognition-Project

# 2. Activate virtual environment (if created)
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

### Application Workflow

```
┌─────────────────────────────────┐
│   Start Application             │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Load Model (best_model.pkl)   │
│   - If found, load it           │
│   - If not found, train new     │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Display Web Interface         │
│   - Input fields for 4 features │
│   - Classify button             │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   User Provides Measurements    │
│   - All 4 values required       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Predict Species               │
│   - Process input features      │
│   - Generate prediction         │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Display Results               │
│   - Predicted species           │
│   - Species images              │
│   - Detailed characteristics    │
└─────────────────────────────────┘
```

### Troubleshooting

**Issue**: Model file not found
- **Solution**: The app will automatically train a new model on first run

**Issue**: Dependencies not found
- **Solution**: Run `pip install -r requirements.txt` again

**Issue**: Port 8501 already in use
- **Solution**: Run `streamlit run app.py --server.port 8502` to use a different port

---

## 👥 Contributing

We welcome contributions to this project! Here's how you can help:

### Types of Contributions
- Bug fixes and error handling improvements
- Feature enhancements (new visualizations, metrics)
- Documentation improvements
- Performance optimizations
- New machine learning models or algorithms
- Test coverage improvements

### Contribution Steps

1. **Fork the Repository**
   ```bash
   # Click the "Fork" button on GitHub
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Write clean, well-documented code
   - Follow Python PEP 8 standards
   - Add comments for complex logic

4. **Commit Your Changes**
   ```bash
   git commit -m "Add: Description of your changes"
   ```

5. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues

### Guidelines
- Maintain code quality and readability
- Test your changes thoroughly
- Document new features
- Update README if necessary

---

## 📄 License

This project is open source and available for educational and research purposes. Users are free to use, modify, and distribute the code according to their needs.

---

## 📚 References

- **Iris Dataset**: Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179-188.
- **scikit-learn Documentation**: https://scikit-learn.org/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Support Vector Machines**: https://en.wikipedia.org/wiki/Support_vector_machine

---

## 📞 Support & Contact

For issues, questions, or suggestions:
- **GitHub Issues**: Open an issue on the GitHub repository
- **Email**: [Contact information if available]

---

## 🎓 Learning Outcomes

By studying this project, you will learn:

1. **Data Science Workflow**: End-to-end machine learning pipeline
2. **Data Preprocessing**: Handling and preparing data for ML
3. **Model Selection**: Evaluating different algorithms
4. **Hyperparameter Tuning**: Optimizing model performance
5. **Model Deployment**: Creating interactive web applications
6. **Web Development**: Using Streamlit for data science apps
7. **Error Handling**: Robust exception management
8. **Best Practices**: Clean code and documentation standards

---

## 🏆 Project Highlights

✅ Complete machine learning pipeline from data to deployment
✅ Interactive web application for real-world predictions
✅ Optimized SVM model with fine-tuned hyperparameters
✅ Comprehensive error handling and validation
✅ Educational resource for ML beginners and practitioners
✅ Well-documented code and clear project structure
✅ Visual feedback with species-specific images

---

**Last Updated**: May 2026  
**Status**: Active and Maintained  
**Repository**: [Iris-Flower-Species-Recognition-Project](https://github.com/Chakrapani2122/Iris-Flower-Species-Recognition-Project)
