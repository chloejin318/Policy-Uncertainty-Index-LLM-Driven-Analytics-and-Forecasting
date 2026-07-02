import requests
import os
import re
# 1.1.1 Data collection: collect articles from Guardian API
GUARDIAN_API_KEY = "e24b790c-4ed1-4e50-9bc9-c59ea17c207b"
GUARDIAN_API_ENDPOINT = "https://content.guardianapis.com/search"

query = 'climate policy OR economic policy OR uncertainty OR carbon price OR carbon emission OR environment policy OR inflation'
section = "environment|business|climate|economy"
order_by = "newest"

output_dir = "guardian_climate_economic_policy_news"
os.makedirs(output_dir, exist_ok=True)

params = {
    "q": query,
    "section": section,
    "order-by": order_by,
    "api-key": GUARDIAN_API_KEY,
    "show-fields": "headline,trailText,bodyText,webUrl",  # Specify fields to retrieve
}

# Make the API request
response = requests.get(GUARDIAN_API_ENDPOINT, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    articles = data.get("response", {}).get("results", [])

    # Save the articles to files
    for i, article in enumerate(articles):
        # Extract and sanitize the title for the filename
        title = article.get("webTitle", f"article_{i}")
        title = re.sub(r"[^\w\-_. ]", "", title)[:100]  # Limit to 100 characters

        # Construct the filename relative to the output directory
        filename = os.path.join(output_dir, f"{title}.txt")

        # Ensure unique filenames
        if os.path.exists(filename):
            filename = os.path.join(output_dir, f"{title}_{i}.txt")

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("Title: " + article.get("webTitle", "No Title") + "\n")
                f.write("Description: " + article.get("fields", {}).get("trailText", "No Description") + "\n")
                f.write("URL: " + article.get("webUrl", "No URL") + "\n")
                f.write("Content:\n" + article.get("fields", {}).get("bodyText", "No Content"))
            print(f"Saved article to {filename}")
        except Exception as e:
            print(f"Error saving article '{title}': {e}")
else:
    print(f"Error fetching news articles: {response.status_code}, {response.text}")

# 1.1.2 Data preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Directory where your collected articles are saved
output_dir = "guardian_climate_economic_policy_news"

# Load the articles from files
articles = []
for filename in os.listdir(output_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(output_dir, filename), "r", encoding="utf-8") as file:
            content = file.read()
            articles.append(content)

# Preprocessing function for text cleaning
def preprocess_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

# Preprocess the articles
processed_articles = [preprocess_text(article) for article in articles]

# Feature extraction (e.g., TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(processed_articles)

# Define the "uncertainty" query to compare articles against
uncertainty_query = "uncertainty climate economic policy inflation carbon emissions environment"

# 1.1.3 Develop LLM-based advanced analysis

# Transform the query using the same vectorizer
uncertainty_query_vec = vectorizer.transform([uncertainty_query])

# Compute cosine similarity between the uncertainty query and each article
cosine_similarities = cosine_similarity(X, uncertainty_query_vec)

# Score calculation based on cosine similarity (higher similarity means higher uncertainty)
uncertainty_scores = cosine_similarities.flatten()

# Standardize the uncertainty scores
scaler = StandardScaler()
uncertainty_scores_standardized = scaler.fit_transform(uncertainty_scores.reshape(-1, 1)).flatten()

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    "Article": os.listdir(output_dir),
    "Uncertainty Score": uncertainty_scores,
    "Standardized Score": uncertainty_scores_standardized
})

# Save the results to a CSV
results_df.to_csv("uncertainty_scores.csv", index=False)
print("Uncertainty scores have been saved.")


import matplotlib.pyplot as plt
# Plot the uncertainty scores
plt.figure(figsize=(10, 6))
plt.hist(uncertainty_scores_standardized, bins=30, alpha=0.7, color='b')
plt.title("Distribution of Standardized Uncertainty Scores")
plt.xlabel("Standardized Uncertainty Score")
plt.ylabel("Frequency")
plt.show()
# Overall, all the above steps are unstructured data corresponding.


# 1.2.2 Implementing the LSTM for forecasting;
# Firstly, we use the LSTM for unstructured data we saved in the csv file
# step 1: Data collection from created csv file

import pandas as pd
from datetime import datetime

output_dir = "guardian_climate_economic_policy_news"
files = os.listdir (output_dir)

# Add a collection date (assuming today's date for the current batch)
collection_date = datetime.today ().strftime ('%Y-%m-%d')

# Initialize a list to store article data
articles_data = []

# Process each article file
for file in files:
    if file.endswith (".txt"):
        filepath = os.path.join (output_dir, file)
        with open (filepath, "r", encoding="utf-8") as f:
            content = f.read ()


        # Extract the uncertainty score for this article (assuming it's calculated already)
        # For this example, we use a placeholder function. Replace with your actual scoring logic.
        def calculate_uncertainty_score(text):
            return len (text.split ()) % 10 / 10.0  # Placeholder: Replace with actual scoring logic


        uncertainty_score = calculate_uncertainty_score (content)

        # Store the article data
        articles_data.append ({
            "date": collection_date,  # Add the collection date here
            "article": content,
            "uncertainty_score": uncertainty_score,
            "standard_uncertainty_score": (uncertainty_score - 0.5) / 0.5,  # Example normalization
        })

# Convert the list of article data to a DataFrame
df_articles = pd.DataFrame (articles_data)

# Save to a CSV file
df_articles.to_csv ("uncertainty_scores.csv", index=False)

print ("Saved articles with uncertainty scores and collection dates to 'uncertainty_scores.csv'")

df_uncertainty = pd.read_csv("uncertainty_scores.csv", parse_dates=["date"])
df_uncertainty.set_index("date", inplace=True)

# Resample the numeric columns only
numeric_columns = df_uncertainty.select_dtypes(include=["number"])
df_uncertainty_resampled = numeric_columns.resample("ME").mean()
print(df_uncertainty_resampled.head())
df_uncertainty_resampled = df_uncertainty_resampled.fillna(method="ffill").fillna(method="bfill")

print(df_uncertainty_resampled.head())
print(df_uncertainty_resampled.info())

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def create_sequences(data, seq_length):
    """
    Create input sequences and corresponding target values for LSTM.

    Args:
        data (np.array): Input data array
        seq_length (int): Number of time steps to use for prediction

    Returns:
        Tuple of input sequences (X) and target values (y)
    """
    x, y = [], []
    for i in range (len (data) - seq_length):
        x.append (data[i:i + seq_length])
        y.append (data[i + seq_length])
    return np.array (x), np.array (y)


def prepare_lstm_forecast(file_path, seq_length=12):
    """
    Prepare and execute LSTM forecasting for uncertainty data.

    Args:
        file_path (str): Path to the CSV file
        seq_length (int): Number of previous time steps to use for prediction

    Returns:
        Tuple of model, predictions, actual values, and performance metrics
    """
    # Load and preprocess the data
    try:
        df_uncertainty = pd.read_csv (file_path, parse_dates=["date"])
        df_uncertainty.set_index ("date", inplace=True)
    except Exception as e:
        print (f"Error loading data: {e}")
        return None

    # Resample to monthly data
    numeric_columns = df_uncertainty.select_dtypes (include=["number"])
    df_uncertainty_resampled = numeric_columns.resample ("M").mean ().fillna (method="ffill").fillna (method="bfill")

    # Scale the data
    scaler = MinMaxScaler (feature_range=(0, 1))
    data_scaled = scaler.fit_transform (df_uncertainty_resampled.values.reshape (-1, 1))

    # Create sequences
    x, y = create_sequences (data_scaled.flatten (), seq_length)

    # Reshape for LSTM input
    x = x.reshape (x.shape[0], x.shape[1], 1)

    # Split into training and testing sets
    train_size = int (len (x) * 0.8)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build LSTM Model
    model = Sequential ([
        LSTM (50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM (50, return_sequences=False),
        Dense (25, activation='relu'),
        Dense (1)
    ])

    # Compile and train the model
    model.compile (optimizer='adam', loss='mean_squared_error')
    model.fit (x_train, y_train, batch_size=32, epochs=50, verbose=1, validation_split=0.2)

    # Make predictions
    predictions = model.predict (x_test)

    # Inverse transform predictions and actual values
    predictions_original = scaler.inverse_transform (predictions)
    y_test_original = scaler.inverse_transform (y_test.reshape (-1, 1))

    # Calculate performance metrics
    mse = mean_squared_error (y_test_original, predictions_original)
    rmse = np.sqrt (mse)

    # Visualize results
    plt.figure (figsize=(12, 6))
    plt.plot (y_test_original, label='Actual Uncertainty', color='blue')
    plt.plot (predictions_original, label='Predicted Uncertainty', color='red', linestyle='--')
    plt.title ('Uncertainty Forecasting using LSTM')
    plt.xlabel ('Time Steps')
    plt.ylabel ('Uncertainty Score')
    plt.legend ()
    plt.tight_layout ()
    plt.show ()

    return {
        'model': model,
        'predictions': predictions_original,
        'actual_values': y_test_original,
        'mse': mse,
        'rmse': rmse
    }


# Usage example
if __name__ == '__main__':
    results = prepare_lstm_forecast ('uncertainty_scores.csv')
    if results:
        print (f"Mean Squared Error: {results['mse']}")
        print (f"Root Mean Squared Error: {results['rmse']}")


# 1.2.3 Implementing the LSTM for forecasting;
# step 1: Integrating with actual structured data from datasets to collect (economic and climate indicators, e.g., GDP, inflation, emissions, etc.)
# Adding Structured Features
# Example structured data (GDP growth, inflation, emissions levels) replace with actual indicator data from world bank
structured_data = pd.DataFrame({
    'gdp_growth': gdp_growth_data,
    'inflation': inflation_data,
    'emissions': emissions_data
})

# Normalize structured data
structured_scaler = MinMaxScaler()
structured_data_scaled = structured_scaler.fit_transform(structured_data)

# Merge structured and uncertainty data (for each time point)
combined_data = np.concatenate([scaled_data, structured_data_scaled], axis=1)

# Create sequences for LSTM with combined data
X_combined, y_combined = create_sequences(combined_data, seq_length=30)

# Reshape X for LSTM (samples, time steps, features)
X_combined = X_combined.reshape(X_combined.shape[0], X_combined.shape[1], X_combined.shape[2])

# Train-test split
X_train_combined, X_test_combined = X_combined[:train_size], X_combined[train_size:]
y_train_combined, y_test_combined = y_combined[:train_size], y_combined[train_size:]

# Build a combined LSTM model
model_combined = tensorflow.keras.models.Sequential()
model_combined.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_combined.shape[1], X_train_combined.shape[2])))
model_combined.add(Dropout(0.2))
model_combined.add(LSTM(units=50, return_sequences=False))
model_combined.add(Dropout(0.2))
model_combined.add(Dense(units=1))

# Compile the model
model_combined.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model_combined.fit(X_train_combined, y_train_combined, epochs=50, batch_size=32, validation_data=(X_test_combined, y_test_combined))

# 1.2.3 Implementing the LSTM for forecasting; step 2: Predict future uncertainty scores
predictions = model_combined.predict(X_test_combined)

# Inverse transform the predictions and actual values
predictions = scaler.inverse_transform(predictions)
y_test_combined = scaler.inverse_transform(y_test_combined)

# Compare predictions to actual values
import matplotlib.pyplot as plt

plt.plot(y_test_combined, label='Actual Uncertainty')
plt.plot(predictions, label='Predicted Uncertainty')
plt.legend()
plt.show()
