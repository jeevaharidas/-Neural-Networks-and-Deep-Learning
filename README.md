# -Neural-Networks-and-Deep-Learning

# 🩻 Chest X-ray Detection Pipeline

A machine learning pipeline to classify chest X-ray images using **Logistic Regression**, **Random Forest**, and **SVM**.  
The pipeline includes image preprocessing, dataset splitting, model training, evaluation, and result visualization.

---

## 📌 Features
- Loads and preprocesses chest X-ray images.
- Supports multiple machine learning models:
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machine (SVM)
- Automated **train-test split**.
- Classification reports with **ROC AUC** and **F1 Score**.
- Saves trained models using `joblib`.
- Visualizes example X-ray images with predictions.

---

## 📂 Project Structure
chest-xray-pipeline/
│── 📄 chest_xray_pipeline.py # Main pipeline script
│── 📄 requirements.txt # Python dependencies
│── 📄 README.md # Project documentation
│── 📂 dataset/ # Dataset folder (contains X-ray images)
│ ├── normal/
│ └── pneumonia/

🚀 Usage
Train and Evaluate Models
python chest_xray_pipeline.py


After running, the script will:

Train Logistic Regression, Random Forest, and SVM models.

Output performance metrics.

Save models as .joblib files.

📊 Example Output
Training and evaluating LogisticRegression...
ROC AUC: 0.9727 | F1 Score: 0.9777

Training and evaluating DecisionTree...
ROC AUC: 0.8541 | F1 Score: 0.9669

Training and evaluating RandomForest...
ROC AUC: 0.9911 | F1 Score: 0.9866

📦 Dependencies

Python 3.8+
OpenCV
NumPy
scikit-learn
matplotlib
Pillow
joblib

Install them via:

pip install -r requirements.txt

🧠 Future Improvements

Add deep learning models (CNNs).

Implement cross-validation for better performance estimation.

Include Grad-CAM or other visualization techniques.
