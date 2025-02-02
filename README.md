# -ML-BASED-PRODUCT-RECOMMENDATION-SYSTEM-FOR-E-COMMERCE-INSIGHTS-RandomForest

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
