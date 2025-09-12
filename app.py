import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from werkzeug.utils import secure_filename
from spark_logic import create_spark_session, engineer_features, detect_anomalies_from_transformed, load_csv_with_fix, transform_features
from helper import allowed_file
import numpy as np

UPLOAD_FOLDER = 'upload'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login-signup.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard/dashboard.html')

@app.route('/analytics')
def analytics():
    return render_template('dashboard/analytics.html')

@app.route('/profile')
def profile():
    return render_template('dashboard/profile.html')

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    # --- Part 1: Robust Input Validation (from the first snippet) ---
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        # --- Part 2: Secure File & Path Management (from the first snippet) ---
        
        # Sanitize the filename to prevent security issues
        filename = secure_filename(file.filename)
        
        # Define input and output paths safely
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Use the output naming convention from the second snippet
        output_filename = filename.rsplit('.', 1)[0] + '_results.csv'
        output_path = os.path.join('static', 'results', output_filename)
        
        # Create directories if they don't exist to prevent errors
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        file.save(input_path)

        try:
            results = runModel(input_path, output_path)

            columns_for_visualization = [
                'Date', 
                'Close', 
                'High', 
                'Low', 
                'Open',
                'Volume',
                'cluster', 
                'distance'
            ]

            clustered_df = results["clustered"]
            anomalies_df = results["anomalies"]

            # --- FIX STARTS HERE: FORMAT THE DATE COLUMN ---
            # Ensure the 'Date' column is a proper datetime object
            clustered_df['Date'] = pd.to_datetime(clustered_df['Date'])
            anomalies_df['Date'] = pd.to_datetime(anomalies_df['Date'])

            # Format the 'Date' column to the 'YYYY-MM-DD' string format
            clustered_df['Date'] = clustered_df['Date'].dt.strftime('%Y-%m-%d')
            anomalies_df['Date'] = anomalies_df['Date'].dt.strftime('%Y-%m-%d')
            # --- FIX ENDS HERE ---

            clustered_for_frontend = results["clustered"][columns_for_visualization]
            anomalies_for_frontend = pd.DataFrame(columns=columns_for_visualization)

            if not results["anomalies"].empty:
                anomalies_for_frontend = results["anomalies"][columns_for_visualization]

            return jsonify({
                "message": "Analysis complete!",
                "output_path": output_path,
                "summary": {
                    "rows": len(results["featured"]),
                    "anomalies": len(results["anomalies"]),
                    "clusters": int(results["clustered"]["cluster"].nunique())
                },
                "results": {
                    "clustered": clustered_for_frontend.to_dict(orient="records"),
                    "anomalies": anomalies_for_frontend.to_dict(orient="records"),
                },
                "threshold": results["threshold"]
            })
            
        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            return jsonify({'error': f'An error occurred during Spark processing: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload a CSV.'}), 400




def runModel(input_path, output_path):
    spark = create_spark_session()
    print("✅ Spark session created.")

    # ---------------- STEP 1: Load Data ----------------
    initial_df = load_csv_with_fix(input_path, spark)
    print("✅ Data loaded.")
    print(f"Schema of initial_df: {initial_df.printSchema()}")
    print("Sample rows from initial_df:")
    initial_df.show(5, truncate=False)

    # ---------------- STEP 2: Feature Engineering ----------------
    featured_df = engineer_features(spark, initial_df)
    print("✅ Feature engineering complete.")
    print("Schema of featured_df:")
    featured_df.printSchema()
    print("Sample rows from featured_df:")
    featured_df.show(5, truncate=False)

    # ---------------- STEP 3: Feature Transformation ----------------
    transformed_df = transform_features(featured_df)
    print("✅ Features transformed.")
    print("Schema of transformed_df:")
    transformed_df.printSchema()
    print("Sample rows from transformed_df:")
    transformed_df.select("Date", "Close", "features").show(5, truncate=False)

    # ---------------- STEP 4: Anomaly Detection ----------------
    df_clustered, anomalies_df, centers, threshold = detect_anomalies_from_transformed(transformed_df)
    print("✅ Anomaly detection complete.")
    print(f"Anomaly threshold: {threshold}")
    print(f"Clustered row count: {df_clustered.count()}, Anomalies count: {anomalies_df.count()}")

    # ---------------- STEP 5: Convert to Pandas ----------------
    featured_pdf = featured_df.toPandas()
    clustered_pdf = df_clustered.toPandas()
    anomalies_pdf = anomalies_df.toPandas()

    print("✅ Converted to Pandas.")
    print(f"featured_pdf shape: {featured_pdf.shape}, columns: {featured_pdf.columns.tolist()}")
    print(f"clustered_pdf shape: {clustered_pdf.shape}, columns: {clustered_pdf.columns.tolist()}")
    print(f"anomalies_pdf shape: {anomalies_pdf.shape}, columns: {anomalies_pdf.columns.tolist()}")

    # ---------------- STEP 6: Save Anomalies ----------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    anomalies_pdf.to_csv(output_path, index=False)
    print(f"✅ Anomalies saved to {output_path}")

    spark.stop()
    print("✅ Spark session stopped.")

    return {
        "featured": featured_pdf,
        "clustered": clustered_pdf,
        "anomalies": anomalies_pdf,
        "centers": centers,
        "threshold": threshold
    }



if __name__ == '__main__':
    app.run(debug=False)