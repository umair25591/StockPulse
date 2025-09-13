import csv
import os
import re
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_from_directory
import pandas as pd
from werkzeug import Response
from werkzeug.utils import secure_filename
from spark import create_spark_session, engineer_features, detect_anomalies_KMeans, load_csv_with_fix, transform_features, detect_anomalies_GMM, select_features, load_csv_pandas, engineer_features_pandas, select_features_pandas
from helper import allowed_file, login_required
import numpy as np
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
import uuid
from bson.objectid import ObjectId
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from collections import Counter
import yfinance as yf

UPLOAD_FOLDER = 'static/uploads'
PROFILE_UPLOAD_FOLDER = "static/uploads/profiles"
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

load_dotenv()

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(PROFILE_UPLOAD_FOLDER, exist_ok=True)


app = Flask(__name__)

app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = os.getenv("SECRET_KEY")

client = MongoClient(os.getenv("MONGO_URI"))
db = client["Stock_Pulse"]
users_collection = db["users"]
history_collection = db["analysis_history"]

bcrypt = Bcrypt(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/blog/isolation-forest-vs-svm')
def blog_isolation_forest_vs_svm():
    return render_template('IsolationForestvs.SVM.html')

@app.route('/blog/volatility-guide')
def blog_volatility_guide():
    return render_template("A Beginner'sGuidetoStockMarketVolatility.html")

@app.route('/blog/myths-debunked')
def blog_myths_debunked():
    return render_template('7CommonStockMarketMythsDebunked.html')

@app.route('/blog/resilient-portfolio')
def blog_resilient_portfolio():
    return render_template('BuildingaResilientPortfolio.html')

@app.route('/blog/candlesticks-guide')
def blog_candlesticks_guide():
    return render_template('DecodingtheCandlesticks.html')

@app.route('/blog/power-of-feature-engineering')
def blog_power_of_feature_engineering():
    return render_template('PowerofFeatureEngineering.html')

@app.route('/blog/spotting-the-signs')
def blog_spotting_the_signs():
    return render_template('SpottingtheSigns.html')

@app.route('/blog/art-of-visualization')
def blog_art_of_visualization():
    return render_template('TheArtofVisualization.html')

@app.route('/blog/future-of-finance')
def blog_future_of_finance():
    return render_template('TheFutureofFinance.html')

@app.route('/blog/unmasking-market-manipulation')
def blog_unmasking_market_manipulation():
    return render_template('UnmaskingMarketManipulation.html')

@app.route('/blog/what-is-algorithmic-trading')
def blog_what_is_algorithmic_trading():
    return render_template('WhatisAlgorithmicTrading.html')

@app.route('/blog/why-data-science-is-your-ally')
def blog_why_data_science_is_your_ally():
    return render_template('WhyDataScienceisYourGreatestAlly.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get("email")
        password = request.form.get("password")

        user = users_collection.find_one({"email": email})

        if user and bcrypt.check_password_hash(user["password"], password):
            # store session data
            session["user_id"] = str(user["_id"])
            session["first_name"] = user["first_name"]
            session["last_name"] = user["last_name"]
            session["role"] = user.get("role", "User")
            session["profile_picture"] = user.get("profile_picture")
            session["background_image"] = user.get("background_image")

            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.", "danger")
            return redirect(url_for("login"))
        
    return render_template('login-signup.html')

@app.route('/signup', methods=['POST'])
def signup():
    first_name = request.form.get("first_name", "").strip()
    last_name = request.form.get("last_name", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password")
    role = request.form.get("role")
    address = request.form.get("address", "").strip()
    profile_picture = request.files.get("profile_picture")

    required_fields = {
        "First Name": first_name, "Last Name": last_name, "Email": email,
        "Password": password, "Role": role
    }
    
    for field_name, value in required_fields.items():
        if not value:
            flash(f"{field_name} is required.", "danger")
            return redirect(url_for("login"))

    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        flash("Invalid email address format.", "danger")
        return redirect(url_for("login"))

    if users_collection.find_one({"email": email}):
        flash("This email address is already registered.", "danger")
        return redirect(url_for("login"))

    picture_filename = None
    if profile_picture and profile_picture.filename:
        if not allowed_file(profile_picture.filename):
            flash("Invalid file format. Please use PNG, JPG, JPEG, or GIF.", "danger")
            return redirect(url_for("login"))
        
        ext = profile_picture.filename.rsplit('.', 1)[1].lower()
        picture_filename = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(PROFILE_UPLOAD_FOLDER, secure_filename(picture_filename))
        profile_picture.save(save_path)

    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')

    users_collection.insert_one({
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "password": hashed_pw,
        "role": role,
        "address": address,
        "profile_picture": picture_filename,
        "created_at": pd.Timestamp.now().isoformat()
    })

    flash("Signup successful! You can now log in.", "success")
    return redirect(url_for("login"))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = ObjectId(session["user_id"])
    
    user_history = list(history_collection.find({"user_id": user_id}))
    
    total_analyses = len(user_history)
    total_anomalies = sum(item['summary'].get('anomalies_found', 0) for item in user_history)
    
    if user_history:
        model_counts = Counter(item['model_used'] for item in user_history)
        most_used_model = model_counts.most_common(1)[0][0] if model_counts else "N/A"
        last_analysis_date = max(item['run_timestamp'] for item in user_history)
    else:
        most_used_model = "N/A"
        last_analysis_date = None

    summary_stats = {
        "total_analyses": total_analyses,
        "total_anomalies": total_anomalies,
        "most_used_model": most_used_model.replace("_", " ").title(),
        "last_analysis_date": last_analysis_date
    }

    recent_analyses = list(history_collection.find({"user_id": user_id})
                           .sort("run_timestamp", -1).limit(5))

    thirty_days_ago = datetime.now() - timedelta(days=30)
    pipeline = [
        {"$match": {"user_id": user_id, "run_timestamp": {"$gte": thirty_days_ago}}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$run_timestamp"}},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": 1}}
    ]
    activity_data = list(history_collection.aggregate(pipeline))
    
    # Prepare data for Chart.js
    chart_labels = [item['_id'] for item in activity_data]
    chart_values = [item['count'] for item in activity_data]

    return render_template('dashboard/dashboard.html', 
                           summary=summary_stats, 
                           recent_analyses=recent_analyses,
                           chart_labels=chart_labels,
                           chart_values=chart_values)

@app.route('/analytics')
@login_required
def analytics():
    return render_template('dashboard/analytics.html')

@app.route('/history')
@login_required
def history():
    user_id = ObjectId(session["user_id"])
    
    user_history = list(history_collection.find({"user_id": user_id})
                        .sort("run_timestamp", -1))
    
    return render_template('dashboard/history.html', history=user_history)

@app.route('/profile')
@login_required
def profile():
    user_id = session.get("user_id")
    user_data = users_collection.find_one({"_id": ObjectId(user_id)})
    
    if not user_data:
        flash("User not found.", "danger")
        return redirect(url_for("logout"))
        
    return render_template('dashboard/profile.html', user=user_data)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    user_id = ObjectId(session["user_id"])

    if request.method == 'POST':
        if 'reset_background' in request.form:
            users_collection.update_one(
                {"_id": user_id},
                {"$unset": {"background_image": ""}}
            )
            session.pop('background_image', None)
            flash('Background has been reset to default.', 'success')
            return redirect(url_for('settings'))

        image_url = request.form.get('backgroundImageUrl')
        if image_url:
            users_collection.update_one(
                {"_id": user_id},
                {"$set": {"background_image": image_url}}
            )
            session['background_image'] = image_url
            flash('Background updated successfully!', 'success')
            return redirect(url_for('settings'))

        if 'backgroundImageFile' in request.files:
            file = request.files['backgroundImageFile']
            if file and file.filename != '' and allowed_file(file.filename):
                ext = file.filename.rsplit('.', 1)[1].lower()
                filename = f"{uuid.uuid4().hex}.{ext}"
                
                upload_path = app.config['UPLOAD_FOLDER']
                if not os.path.exists(upload_path):
                    os.makedirs(upload_path)
                
                file.save(os.path.join(upload_path, secure_filename(filename)))
                
                users_collection.update_one(
                    {"_id": user_id},
                    {"$set": {"background_image": filename}}
                )
                
                session['background_image'] = filename
                
                flash('Background updated successfully!', 'success')
                return redirect(url_for('settings'))
            elif file.filename != '':
                flash('Invalid file type. Please upload an image (png, jpg, gif).', 'danger')
                return redirect(url_for('settings'))

    return render_template('dashboard/settings.html')

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    user_id = session.get("user_id")
    
    first_name = request.form.get("first_name")
    last_name = request.form.get("last_name")
    email = request.form.get("email")
    address = request.form.get("address")
    role = request.form.get("role")
    
    bio = request.form.get("bio")
    twitter_url = request.form.get("twitter_url")
    linkedin_url = request.form.get("linkedin_url")
    github_url = request.form.get("github_url")

    update_data = {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "address": address,
        "role": role,
        "bio": bio,
        "socials": {
            "twitter": twitter_url,
            "linkedin": linkedin_url,
            "github": github_url,
        }
    }

    users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": update_data}
    )
    
    session["first_name"] = first_name
    session["last_name"] = last_name
    
    flash("Profile updated successfully!", "success")
    return redirect(url_for('profile'))

@app.route('/run_analysis', methods=['POST'])
@login_required
def run_analysis():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    stock_symbol = request.form.get('stock_symbol', 'UNKNOWN').upper()
    if not stock_symbol:
        return jsonify({'error': 'Stock Ticker is a required field.'}), 400

    model_choice = request.form.get('model', 'kmeans')
    
    try:
        threshold = float(request.form.get('threshold', 3.0))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid threshold value provided.'}), 400

    if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() == 'csv':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        original_filename = secure_filename(file.filename)
        input_filename = f"{timestamp}_{original_filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        
        file.save(input_path)

        expected_columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        try:
            uploaded_header = pd.read_csv(input_path, nrows=0).columns.tolist()
            
            if uploaded_header != expected_columns:
                os.remove(input_path)
                error_message = (
                    "Invalid CSV format. The header is incorrect. "
                    f"Please ensure the columns are exactly: {', '.join(expected_columns)}"
                )
                return jsonify({'error': error_message}), 400

        except Exception as e:
            os.remove(input_path)
            app.logger.error(f"Error validating CSV header for {original_filename}: {e}")
            return jsonify({
                'error': 'The uploaded file could not be read. Please ensure it is a valid, uncorrupted CSV.'
            }), 400

        try:
            results = runModel(
                input_path=input_path, 
                model_choice=model_choice, 
                threshold=threshold
            )

            if not results or "clustered" not in results or "anomalies" not in results:
                return jsonify({'error': 'Model execution failed to produce valid results.'}), 500

            clustered_df = results["clustered"]
            anomalies_df = results["anomalies"]
            
            final_df_for_download = clustered_df.copy()
            final_df_for_download['is_anomaly'] = 0
            final_df_for_download.loc[final_df_for_download.index.isin(anomalies_df.index), 'is_anomaly'] = 1
            final_df_for_download['model_used'] = model_choice
            
            download_filename = f"{timestamp}_{model_choice}_results.csv"
            download_path = os.path.join(app.config['RESULTS_FOLDER'], download_filename)
            final_df_for_download.to_csv(download_path, index=False)
            
            try:
                history_document = {
                    "user_id": ObjectId(session["user_id"]),
                    "run_timestamp": datetime.now(),
                    "original_filename": original_filename,
                    "stock_symbol": stock_symbol,
                    "model_used": model_choice,
                    "parameters": { "threshold_multiplier": threshold },
                    "summary": {
                        "rows_processed": len(clustered_df),
                        "anomalies_found": len(anomalies_df)
                    },
                    "results_filename": download_filename
                }
                history_collection.insert_one(history_document)
                
            except Exception as e:
                app.logger.error(f"Failed to save analysis to MongoDB history: {e}")

            for df in [clustered_df, anomalies_df]:
                if not df.empty and 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

            columns_for_visualization = [
                'Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'cluster', 'distance'
            ]

            def safe_select_columns(df, columns):
                existing_cols = [col for col in columns if col in df.columns]
                return df[existing_cols]

            clustered_for_frontend = safe_select_columns(clustered_df, columns_for_visualization)
            anomalies_for_frontend = safe_select_columns(anomalies_df, columns_for_visualization)

            return jsonify({
                "message": "Analysis complete!",
                "summary": {
                    "rows": len(clustered_df),
                    "anomalies": len(anomalies_df),
                    "clusters": int(clustered_df["cluster"].nunique()) if not clustered_df.empty and 'cluster' in clustered_df.columns else 0
                },
                "results": {
                    "clustered": clustered_for_frontend.to_dict(orient="records"),
                    "anomalies": anomalies_for_frontend.to_dict(orient="records"),
                },
                "threshold": results.get("threshold", threshold),
                "download_filename": download_filename,
                "stock_symbol": stock_symbol
            })
            
        except Exception as e:
            app.logger.error(f"An error occurred during analysis for {original_filename}: {e}", exc_info=True)
            return jsonify({'error': f'An error occurred during data processing: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload a CSV.'}), 400

@app.route('/download_results/<filename>')
@login_required
def download_results(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)

@app.route('/data_downloader')
@login_required
def data_downloader():
    return render_template('dashboard/data_downloader.html')

@app.route('/download_yfinance', methods=['POST'])
@login_required
def download_yfinance():
    try:
        ticker = request.form.get('ticker').upper()
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        if not all([ticker, start_date, end_date]):
            flash("All fields are required.", "danger")
            return redirect(url_for('data_downloader'))

        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            flash(f"No data found for ticker '{ticker}' in the specified date range.", "warning")
            return redirect(url_for('data_downloader'))

        if len(data) > 2:
            data = data.iloc[2:]
        else:
            flash(f"Not enough data for ticker '{ticker}' to remove 2 rows.", "warning")
            return redirect(url_for('data_downloader'))

        data.reset_index(inplace=True)

        temp_df = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        temp_df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        final_df = temp_df

        csv_string = final_df.to_csv(index=False)

        return Response(
            csv_string,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename={ticker}_{start_date}_to_{end_date}.csv"}
        )

    except Exception as e:
        flash(f"An error occurred: {e}", "danger")
        return redirect(url_for('data_downloader'))
    
def runModel(input_path, model_choice='kmeans', threshold=3.0):
    
    if model_choice in ['kmeans', 'gmm']:
        spark = create_spark_session()
        
        initial_df = load_csv_with_fix(input_path, spark)
        featured_df = engineer_features(spark, initial_df)
        clean_featured_df = featured_df.dropna()
        
        selected_features = select_features(clean_featured_df)
        transformed_df = transform_features(clean_featured_df, selected_features)

        if model_choice == 'kmeans':
            df_clustered, anomalies_df, centers, threshold_value = detect_anomalies_KMeans(
                transformed_df, threshold_std_dev=threshold
            )
        elif model_choice == 'gmm':
            df_clustered, anomalies_df, centers, threshold_value = detect_anomalies_GMM(
                transformed_df, threshold_std_dev=threshold
            )
        
        clustered_pdf = df_clustered.toPandas()
        anomalies_pdf = anomalies_df.toPandas()
        
        spark.stop()

    elif model_choice in ['isolation_forest', 'svm']:
        
        pdf = load_csv_pandas(input_path)
        featured_pdf = engineer_features_pandas(pdf)
        selected_features = select_features_pandas(featured_pdf)
        
        scaler = StandardScaler()
        features = scaler.fit_transform(featured_pdf[selected_features])

        if model_choice == 'isolation_forest':
            model = IsolationForest(contamination=0.05, random_state=42)
            predictions = model.fit_predict(features) 
        
        elif model_choice == 'svm':
            model = OneClassSVM(nu=0.05)
            predictions = model.fit_predict(features)

        featured_pdf['anomaly'] = predictions
        anomalies_pdf = featured_pdf[featured_pdf['anomaly'] == -1].copy()
        clustered_pdf = featured_pdf.copy() 

        anomalies_pdf['distance'], clustered_pdf['distance'] = 0, 0
        anomalies_pdf['cluster'], clustered_pdf['cluster'] = -1, 0
        centers, threshold_value = None, "N/A (scikit-learn)"

    else:
        raise ValueError(f"Unsupported model type: '{model_choice}'")

    return {
        "clustered": clustered_pdf,
        "anomalies": anomalies_pdf,
        "centers": centers,
        "threshold": threshold_value
    }

if __name__ == '__main__':
    app.run(debug=True)