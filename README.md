# House Category Prediction  

## 📽 Video Demo  

Check out the video demonstration of the project: [Watch on YouTube](https://youtu.be/2VmEhazhHIk)).  

---

##  Project Repository  

Access the repository here: [House Prediction GitHub Repo](https://github.com/NdanyuzweP/House_Category).  
Access the deployed model and API here: https://house-category-1.onrender.com/doc

Also there is Dockerfile with docker image

---

## Project Description  

The **House Category Prediction** project leverages machine learning to predict house Category based on various features such as size, location, and other relevant attributes. This project demonstrates:  

- **Data preprocessing**: Cleaning and preparing raw data for modeling.  
- **Model training**: Implementing and optimizing a machine learning algorithm for Category prediction.  
- **Evaluation**: Assessing the model’s performance and accuracy using metrics such as Mean Squared Error (MSE).  
- **Deployment**: Making predictions accessible through a user-friendly interface.  

This project is ideal for understanding the workflow of a data science and machine learning pipeline.  

---

## 🛠️ Setup Instructions  

### Prerequisites  
Ensure the following tools and dependencies are installed:  

- **Python** (>= 3.7)  
- Python libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`  

### Steps to Set Up  

1. Clone the repository:  
   ```bash
   git clone https://github.com/NdanyuzweP/House_Category.git
   cd House_Category

2. Set up a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'

3. Install required dependencies
   ```bash
   pip install -r requirements.txt

4. Run the application or model training script
   ```bash
   uvicorn main:app --reload --port 8080


### Steps to Run Frontend

1. Clone the repository:  
   ```bash
   git clone https://github.com/NdanyuzweP/House_Category.git
   cd House_Category
   cd frontend/house_predictions
   cd src

2. run react app
   ```bash
   npm start
   



### Steps to Run Backend (API)

1. Clone the repository:  
   ```bash
   git clone https://github.com/NdanyuzweP/House_Category.git
   cd House_Category

2. Set run fastapi
   ```bash
   uvicorn main:app --reload --port 8080



