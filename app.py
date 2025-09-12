import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from werkzeug.utils import secure_filename
from spark_logic import create_spark_session, engineer_features, detect_anomalies_KMeans, load_csv_with_fix, transform_features, detect_anomalies_GMM
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
    # --- Part 1: Robust Input Validation ---
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # --- CHANGE #1: GET USER PARAMETERS FROM THE REQUEST FORM ---
    # Get the selected model, defaulting to 'kmeans' if not provided
    model_choice = request.form.get('model', 'kmeans')
    
    # Get the threshold, defaulting to 3.0. Includes error handling for non-numeric values.
    try:
        threshold = float(request.form.get('threshold', 3.0))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid threshold value provided.'}), 400

    if file and allowed_file(file.filename): # Make sure allowed_file function exists
        # --- Part 2: Secure File & Path Management ---
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # --- CHANGE #2: MAKE OUTPUT FILENAME MORE DESCRIPTIVE ---
        output_filename = f"{filename.rsplit('.', 1)[0]}_{model_choice}_results.csv"
        output_path = os.path.join(app.config.get('RESULTS_FOLDER', 'static/results'), output_filename)
        
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        file.save(input_path)

        try:
            # --- CHANGE #3: PASS NEW PARAMETERS TO YOUR MODEL FUNCTION ---
            # IMPORTANT: Your runModel function must be updated to accept these new arguments
            results = runModel(
                input_path=input_path, 
                output_path=output_path, 
                model_choice=model_choice, 
                threshold=threshold
            )

            if not results or "clustered" not in results or "anomalies" not in results:
                return jsonify({'error': 'Model execution failed to produce valid results.'}), 500

            clustered_df = results["clustered"]
            anomalies_df = results["anomalies"]

            # Format the 'Date' column for consistent JSON serialization
            for df in [clustered_df, anomalies_df]:
                if not df.empty and 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

            # --- CHANGE #4: SAFER COLUMN SELECTION FOR ROBUSTNESS ---
            columns_for_visualization = [
                'Date', 'Close', 'High', 'Low', 'Open',
                'Volume', 'cluster', 'distance'
            ]

            # Helper to select only columns that actually exist in a dataframe
            def safe_select_columns(df, columns):
                existing_cols = [col for col in columns if col in df.columns]
                return df[existing_cols]

            clustered_for_frontend = safe_select_columns(clustered_df, columns_for_visualization)
            anomalies_for_frontend = safe_select_columns(anomalies_df, columns_for_visualization)

            return jsonify({
                "message": "Analysis complete!",
                "summary": {
                    "rows": len(results.get("featured", clustered_df)),
                    "anomalies": len(anomalies_df),
                    "clusters": int(clustered_df["cluster"].nunique()) if not clustered_df.empty and 'cluster' in clustered_df.columns else 0
                },
                "results": {
                    "clustered": clustered_for_frontend.to_dict(orient="records"),
                    "anomalies": anomalies_for_frontend.to_dict(orient="records"),
                },
                "threshold": results.get("threshold", threshold) # Return the threshold used
            })
            
        except Exception as e:
            # Using app.logger provides better debugging information in your server logs
            app.logger.error(f"An error occurred during analysis for {filename}: {e}", exc_info=True)
            return jsonify({'error': f'An error occurred during data processing: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload a CSV.'}), 400


def runModel(input_path, output_path, model_choice='kmeans', threshold=3.0):
    """
    Main analysis pipeline that dynamically selects the clustering model.
    """
    spark = create_spark_session()
    print("✅ Spark session created.")

    # ---------------- STEP 1, 2, 3: Data Prep (No changes needed) ----------------
    initial_df = load_csv_with_fix(input_path, spark)
    featured_df = engineer_features(spark, initial_df)
    transformed_df = transform_features(featured_df)
    print("✅ Data loading and feature engineering complete.")
    
    # ---------------- STEP 4: DYNAMIC Anomaly Detection ----------------
    print(f"▶️ Running analysis with model: '{model_choice}' and threshold multiplier: {threshold}")
    
    if model_choice == 'kmeans':
        df_clustered, anomalies_df, centers, threshold_value = detect_anomalies_KMeans(
            transformed_df, 
            threshold_std_dev=threshold
        )
    elif model_choice == 'gmm':
        df_clustered, anomalies_df, centers, threshold_value = detect_anomalies_GMM(
            transformed_df,
            threshold_std_dev=threshold
        )
    else:
        spark.stop()
        raise ValueError(f"Unsupported model type: '{model_choice}'. Please choose 'kmeans' or 'gmm'.")

    print("✅ Anomaly detection complete.")
    print(f"   - Anomaly threshold calculated: {threshold_value:.4f}")
    print(f"   - Clustered row count: {df_clustered.count()}, Anomalies count: {anomalies_df.count()}")

    # ---------------- STEP 5: Convert to Pandas ----------------
    featured_pdf = featured_df.toPandas()
    clustered_pdf = df_clustered.toPandas()
    anomalies_pdf = anomalies_df.toPandas()
    print("✅ Converted results to Pandas.")

    # ---------------- STEP 6: Save Anomalies ----------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    anomalies_pdf.to_csv(output_path, index=False)
    print(f"✅ Anomalies saved to {output_path}")

    spark.stop()
    print("✅ Spark session stopped.")

    # Return the results dictionary
    return {
        "featured": featured_pdf,
        "clustered": clustered_pdf,
        "anomalies": anomalies_pdf,
        "centers": centers,
        "threshold": threshold_value # Return the actual calculated threshold
    }

if __name__ == '__main__':
    app.run(debug=False)