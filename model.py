#ML-BASED PRODUCT RECOMMENDATION SYSTEM FOR E-COMMERCE INSIGHTS-RandomForest

#Problem Definition: This system aims to recommend products to users based on their preferences, which are derived from product features like category, price, rating, and number of ratings. By using machine learning techniques, specifically Random Forest Regressor, the system predicts the ratings of products and filters them based on user-defined criteria such as minimum rating and maximum price. The goal is to provide personalized product recommendations to enhance the shopping experience, allowing users to easily discover products that best match their needs.

#Use Cases:
#Suggest products to users based on their selected category (e.g., "jeans").
#Recommend products to users based on price range and minimum rating preferences.
#Improve the shopping experience by helping users find the best products according to their specific preferences.
#Provide a data-driven recommendation system that learns from historical product features to suggest new and relevant products.

#Expected Outcome:
#The system will display a list of recommended products that meet the user's criteria (category, rating, price).
#Users will receive personalized suggestions based on their interactions and input preferences, improving overall customer satisfaction and engagement.

#Libraries Used:
#Pandas for data manipulation and cleaning
#Scikit-learn for machine learning models (Random Forest Regressor)
#Flask for creating a real-time recommendation API
#LabelEncoder for converting categorical product names to numerical values
#Mean Squared Error (MSE) for evaluating the performance of the recommendation model

#Features of the System:
#Content-Based Filtering: Recommends products based on product features such as category, price, ratings, and discount price.
#Personalized Recommendations: Uses machine learning (Random Forest) to predict ratings and recommend products based on user-specified criteria (rating, price range).
#Real-Time Recommendations: A simple API built using Flask that allows users to interact with the system and get real-time product recommendations.
#Easy to Use: Users can filter products by category and further refine recommendations based on ratings and price.

#How to Run:
#Install the required libraries:
#pip install pandas scikit-learn flask

#Run the script to generate recommendations:
#python recommendation_system.py

#Conclusion:
#This recommendation system enhances the shopping experience by suggesting products that match the user's preferences, based on their specified category, rating, and price range. The system uses a machine learning model for prediction and can be deployed in real-time through a simple API, making it scalable and easy to integrate into an e-commerce platform.

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Load dataset
file_path = 'D:/Recommendation System/Amazon_Products_Data.csv'
df = pd.read_csv(file_path)

# Handle missing values
# Check for missing values
df.isnull().sum()

df=df.dropna()

selected_features = ['products','main_category','sub_category','ratings','no_of_ratings','discount_price','actual_price']
df = df[selected_features]

df.head()

# Preprocessing
df['ratings'] = df['ratings'].astype(float)
df['no_of_ratings'] = df['no_of_ratings'].astype(float)
df['discount_price'] = df['discount_price'].astype(float)
df['actual_price'] = df['actual_price'].astype(float)

# Step 1: Prepare the data for training and testing
# Use 'discount_price', 'actual_price', and 'no_of_ratings' as features to predict ratings
X = df[['discount_price', 'actual_price', 'no_of_ratings']]
# Step 2: Label Encoding for the product names (target variable)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['products'])

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 3: Train a Random Forest model
model = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

df['predicted_ratings'] = model.predict(X)


# User input for product category
product_category = input("Enter the product category you are looking for (e.g., jeans): ").lower()

#Filter products based on the category (searching for products containing the category keyword in the name)
filtered_products = df[df['products'].str.contains(product_category, case=False, na=False)]

if filtered_products.empty:
    print(f"No products found for the category: {product_category}")
else:
    # Step 3: Show all products related to the search (before filtering by rating and price)
    print(f"\nRecommended Products based on your criteria (first 75 results) '{product_category}':")
    print(filtered_products[['products', 'ratings', 'discount_price', 'actual_price']])

    # Step 4: Ask for user input for recommendation criteria (min rating and max price)
    min_rating = float(input("\nEnter the minimum rating you are looking for (e.g., 4.0): "))
    max_price = float(input("Enter the maximum price you are looking for (e.g., 3000): "))

    # Step 5: Filter products based on the rating and price range
    filtered_products = filtered_products[
        (filtered_products['ratings'] >= min_rating) &
        (filtered_products['discount_price'] <= max_price)
    ]

    if filtered_products.empty:
        print(f"\nNo products match the criteria for category '{product_category}' with a maximum price of {max_price}.")
    else:
        # Step 6: Display the filtered and recommended products
        print(f"\nRecommended Products based on your criteria:")
        print(filtered_products[['products', 'ratings', 'discount_price', 'actual_price']])
