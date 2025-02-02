from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Initialize Flask app
app = Flask(__name__)

# Load dataset and train model
file_path = 'D:/Recommendation System/Amazon_Products_Data.csv'
df = pd.read_csv(file_path)

# Handle missing values
df = df.dropna()

selected_features = ['products', 'main_category', 'sub_category', 'ratings', 'no_of_ratings', 'discount_price', 'actual_price', 'image']
df = df[selected_features]

# Preprocessing
df['ratings'] = df['ratings'].astype(float)
df['no_of_ratings'] = df['no_of_ratings'].astype(float)
df['discount_price'] = df['discount_price'].astype(float)
df['actual_price'] = df['actual_price'].astype(float)

# Prepare data for training
X = df[['discount_price', 'actual_price', 'no_of_ratings']]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['products'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict ratings
df['predicted_ratings'] = model.predict(X)

@app.route('/')
def index():
    return render_template('index.html', products=[], stage='initial')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get product category from user input
    product_category = request.form['product_category'].lower()
    filtered_products = df[df['products'].str.contains(product_category, case=False, na=False)]

    # Limit to 70 results
    recommended_products = filtered_products.head(75).to_dict(orient='records')

    if not recommended_products:
        message = f"No products found for the category: {product_category}"
        return render_template('index.html', products=[], message=message, stage='initial')

    # Pass recommended products to the next stage
    return render_template(
        'index.html',
        products=recommended_products,
        product_category=product_category,
        stage='filter'
    )

@app.route('/filter', methods=['POST'])
def filter_products():
    # Retrieve form inputs
    product_category = request.form['product_category']
    min_rating = float(request.form['min_rating'])
    max_price = float(request.form['max_price'])

    # Filter products based on rating and price range
    filtered_products = df[
        (df['products'].str.contains(product_category, case=False, na=False)) &
        (df['ratings'] >= min_rating) &
        (df['discount_price'] <= max_price)
    ]

    # Show top 75 filtered results
    top_75_filtered = filtered_products.head(75)
    return render_template('result.html', products=top_75_filtered, step='filtered')


if __name__ == '__main__':
    app.run(debug=True)
