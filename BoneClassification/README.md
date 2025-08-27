# 🦴 Bone Fracture Classification using Machine Learning  

## 📌 Project Overview  
This project implements a **Random Forest classifier** to classify different types of bone X-ray images (Chest, Elbow, Finger, Hand, Head, Shoulder, Wrist). The model is trained on extracted image features and helps in identifying bone types to assist in fracture detection.  

## ⚙️ Features  
- **Dataset Collection/Upload** → Load bone X-ray dataset (e.g., MURA dataset).  
- **Feature Extraction** → Extract RGB pixel values and build feature vectors.  
- **Train & Test Data Split** → Automatically split dataset (80% training, 20% testing).  
- **Random Forest Model Training** → Train Random Forest classifier on extracted features.  
- **Bone Classification** → Upload a test image and classify it into one of the seven bone categories.  

## 🗂️ Dataset  
- The project uses the **MURA Bone X-ray dataset**.  
- Dataset structure:  
  ```
  Dataset/
  ├── Chest/
  ├── Elbow/
  ├── Finger/
  ├── Hand/
  ├── Head/
  ├── Shoulder/
  └── Wrist/
  ```  

## 🖥️ Installation & Setup  
### Requirements  
- Python 3.x  
- Libraries:  
  ```
  numpy  
  pandas  
  scikit-learn  
  opencv-python  
  matplotlib  
  tkinter  
  pickle  
  ```  

### Steps to Run  
1. Clone or download this repository.  
2. Place your dataset in the **Dataset/** folder.  
3. Run the project using:  
   ```bash
   run.bat
   ```  
   *(or run `python BoneClassification.py` directly)*  

## 🎮 Usage  
1. **Upload Dataset** → Load bone dataset into the application.  
2. **Extract Features** → Convert images into feature vectors.  
3. **Train & Test Split** → Prepare training/testing sets.  
4. **Train Random Forest** → Build the classification model.  
5. **Upload Test Image** → Get classification result with predicted bone type.  

## 📊 Results  
- Achieved around **84% accuracy** on the test dataset.  
- Classification report includes **precision, recall, and F1-score** for each bone type.  

## 📸 Screenshots  
Refer to **SCREENS.docx** for detailed step-by-step screenshots.  

## 📌 Future Enhancements  
- Extend dataset with more bone types.  
- Use advanced deep learning models (CNNs) for higher accuracy.  
- Deploy as a web or desktop application for medical use.  
