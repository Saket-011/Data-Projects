import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from bs4 import BeautifulSoup
import requests

# Data Collection and Preprocessing for JEE Rank Predictor

# Assuming you have a CSV file containing historical JEE data
jee_data = pd.read_csv("jee_data.csv")

# Preprocessing
# Drop rows with missing values
jee_data.dropna(inplace=True)

# Encode categorical variables
jee_data = pd.get_dummies(jee_data, columns=['gender', 'location'])

# Split data into features (X) and target (y)
X = jee_data.drop(columns=['rank'])
y = jee_data['rank']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline for preprocessing and model training
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize numerical features
    ('regressor', RandomForestRegressor())  # RandomForestRegressor as an example model
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)


# College Prediction

# Scraping college data from a website
college_data = []

# Example scraping code (using BeautifulSoup)
url = "example.com/colleges"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract relevant information
for college in soup.find_all('div', class_='college'):
    name = college.find('h2').text
    cutoff_rank = college.find('span', class_='cutoff-rank').text
    location = college.find('span', class_='location').text
    # Add data to the list
    college_data.append({'name': name, 'cutoff_rank': cutoff_rank, 'location': location})

# Convert scraped data to DataFrame
college_df = pd.DataFrame(college_data)

# Preprocessing
# Assuming 'cutoff_rank' needs to be converted to numerical format
college_df['cutoff_rank'] = college_df['cutoff_rank'].astype(int)

# Make predictions for colleges based on JEE rank
# Assuming you have a function to predict rank given JEE scores
def predict_rank(scores):
    return pipeline.predict(scores.reshape(1, -1))

# Assuming you have JEE scores for a student
student_scores = [80, 85, 90, 18, 0, 1, 0, 0]  # Example scores

# Predict rank for the student
predicted_rank = predict_rank(student_scores)

# Filter colleges based on predicted rank
eligible_colleges = college_df[college_df['cutoff_rank'] >= predicted_rank[0]]

# Display recommended colleges
print("Recommended Colleges based on Predicted Rank:")
print(eligible_colleges)
