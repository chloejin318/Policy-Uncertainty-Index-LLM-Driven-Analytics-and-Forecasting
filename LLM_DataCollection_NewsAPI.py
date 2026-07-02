import requests
import os
import re

NEWS_API_KEY = "a2cea4ace92844fda071ebcee60b1865"
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

query = "climate policy OR economic policy OR uncertainty OR carbon price OR carbon emission OR environment policy OR inflation"
language = "en"
sort_by = "publishedAt"

output_dir = "C:\\Users\\Chloe\\PycharmProjects\\pythonProject"
os.makedirs(output_dir, exist_ok=True)

# Make the API request
params = {"q": query, "language": language, "sortBy": sort_by, "apiKey": NEWS_API_KEY}
response = requests.get(NEWS_API_ENDPOINT, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Extract the article data
    articles = response.json().get("articles", [])

    # Save the articles to files
    for i, article in enumerate(articles):
        # Remove invalid characters from the title and truncate
        title = article.get(
            "title", f"article_{i}"
        )  # Fallback to a generic name if no title
        title = re.sub(r"[^\w\-_\. ]", "", title)[:100]  # Limit to 100 characters

        # Construct the filename relative to the output directory
        filename = os.path.join(output_dir, f"{title}.txt")

        # Ensure unique filenames in case of duplicates
        if os.path.exists(filename):
            filename = os.path.join(output_dir, f"{title}_{i}.txt")

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(article.get("title", "No Title") + "\n")
                f.write(article.get("description", "No Description") + "\n")
                f.write(article.get("url", "No URL") + "\n")
                f.write(article.get("content", "No Content"))
            print(f"Saved article to {filename}")
        except Exception as e:
            print(f"Error saving article '{title}': {e}")
else:
    print(f"Error fetching news articles: {response.status_code}")
