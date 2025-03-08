from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

app = Flask(__name__)

# Load dataset (replace with actual dataset path)
df = pd.read_csv("Crop_production.csv")

# Encode categorical Crop column
label_encoder = LabelEncoder()
df['Crop_encoded'] = label_encoder.fit_transform(df['Crop'])

# Define features for comparison
features = ['N', 'P', 'K', 'pH', 'rainfall', 'temperature']
X = df[features].values  # Convert to NumPy array

# Apply feature scaling for better distance calculations
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Nearest Neighbors model
nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto').fit(X_scaled)


def get_high_yield_crops(N, P, K, pH, rainfall, temperature, min_crops=10, yield_percentile=50):
    try:
        input_features = scaler.transform([[N, P, K, pH, rainfall, temperature]])  # Apply scaling

        # Find similar condition crops
        distances, indices = nbrs.kneighbors(input_features, n_neighbors=50)

        # Extract matching crop data
        similar_crops_df = df.iloc[indices[0]].copy()

        # Adaptive yield threshold
        current_percentile = yield_percentile
        high_yield_similar = pd.DataFrame()

        while current_percentile > 0:
            yield_threshold = similar_crops_df['Yield_ton_per_hec'].quantile(current_percentile / 100)
            high_yield_similar = similar_crops_df[similar_crops_df['Yield_ton_per_hec'] >= yield_threshold]
            if len(high_yield_similar['Crop'].unique()) >= min_crops:
                break
            current_percentile -= 5  # Reduce threshold dynamically

        # If still less than min_crops, pick the closest condition matches
        if len(high_yield_similar['Crop'].unique()) < min_crops:
            similar_crops_df['condition_distance'] = similar_crops_df[features].apply(
                lambda row: np.sqrt(sum((row - [N, P, K, pH, rainfall, temperature])**2)), axis=1)
            similar_crops_df = similar_crops_df.sort_values('condition_distance')
            high_yield_similar = similar_crops_df.head(min_crops)

        # Additional fallback: Add highest-yielding crops if necessary
        if len(high_yield_similar['Crop'].unique()) < min_crops:
            top_yield_crops = df.sort_values('Yield_ton_per_hec', ascending=False)['Crop'].unique()
            additional_crops = [crop for crop in top_yield_crops if crop not in high_yield_similar['Crop'].values]
            additional_crops = additional_crops[:min_crops - len(high_yield_similar['Crop'].unique())]

            for crop in additional_crops:
                crop_data = df[df['Crop'] == crop]
                high_yield_similar = pd.concat([high_yield_similar, crop_data], ignore_index=True)

        # Compute similarity scores
        result = high_yield_similar.groupby('Crop').agg({
            'Yield_ton_per_hec': 'mean',
            'N': lambda x: abs(x.mean() - N),
            'P': lambda x: abs(x.mean() - P),
            'K': lambda x: abs(x.mean() - K),
            'pH': lambda x: abs(x.mean() - pH),
            'rainfall': lambda x: abs(x.mean() - rainfall),
            'temperature': lambda x: abs(x.mean() - temperature)
        }).reset_index()

        # Normalize yield score
        yield_max, yield_min = result['Yield_ton_per_hec'].max(), result['Yield_ton_per_hec'].min()
        result['yield_score'] = (result['Yield_ton_per_hec'] - yield_min) / (yield_max - yield_min + 1e-6)

        # Normalize feature matching scores
        for col in features:
            col_max, col_min = result[col].max(), result[col].min()
            result[f'{col}_score'] = 1.0 if col_max == col_min else 1 - ((result[col] - col_min) / (col_max - col_min))

        # Compute composite score
        condition_cols = [f'{col}_score' for col in features]
        result['condition_score'] = result[condition_cols].mean(axis=1)
        result['composite_score'] = 0.8 * result['condition_score'] + 0.2 * result['yield_score']

        # Sort by scores
        result = result.sort_values(['composite_score', 'condition_score', 'yield_score'], ascending=[False, False, False])

        # Return as a sorted list rather than a dictionary to preserve order
        recommendations = []
        for _, row in result.iterrows():
            crop_name = row['Crop']
            recommendations.append({
                'crop': crop_name,
                'yield_ton_per_hec': round(row['Yield_ton_per_hec'], 2),
                'composite_score': round(row['composite_score'], 2),
                'yield_score': round(row['yield_score'], 2),
                'condition_match': round(row['condition_score'], 2)
            })
            if len(recommendations) >= min_crops:
                break

        return {'recommendations': recommendations}

    except Exception as e:
        return {"error": str(e)}


@app.route('/predict', methods=['POST'])
def high_yield_crops():
    try:
        data = request.get_json()

        # Validate input values
        N = float(data.get('N', 0))
        P = float(data.get('P', 0))
        K = float(data.get('K', 0))
        pH = float(data.get('pH', 0))
        rainfall = float(data.get('rainfall', 0))
        temperature = float(data.get('temperature', 0))
        min_crops = int(data.get('min_crops', 10))

        # Check valid range (adjust as per your dataset)
        if not (0 <= N <= 200 and 0 <= P <= 200 and 0 <= K <= 200):
            return jsonify({'error': 'N, P, and K values must be between 0 and 200'}), 400
        if not (4 <= pH <= 9):
            return jsonify({'error': 'pH value must be between 4 and 9'}), 400
        if not (0 <= rainfall <= 5000):
            return jsonify({'error': 'Rainfall must be between 0 and 5000 mm'}), 400
        if not (-10 <= temperature <= 50):
            return jsonify({'error': 'Temperature must be between -10 and 50°C'}), 400

        # Get recommendations
        recommendations = get_high_yield_crops(N, P, K, pH, rainfall, temperature, min_crops=min_crops)

        return jsonify(recommendations)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)