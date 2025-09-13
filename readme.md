# 📈 StockPulse: Stock Market Anomaly Detection

**StockPulse** is a web-based application designed for analyzing stock market data to identify anomalies and unusual trading patterns.  
It leverages **Apache Spark** for large-scale data processing and employs various **machine learning models** to detect potential market irregularities.  
The application is built with **Flask**, providing a user-friendly interface for uploading data, running analyses, and visualizing results.

---

## 🚀 Features

- 📊 **Data Analysis**: Upload your own stock data in CSV format.  
- 🤖 **Multiple ML Models**: Choose from KMeans, Gaussian Mixture Models (GMM), Isolation Forest, and One-Class SVM for anomaly detection.  
- 📂 **Interactive Dashboard**: View analysis history, summary statistics, and user activity.  
- 📈 **Data Visualization**: Interactive charts to visualize clustered data and identified anomalies.  
- 🔐 **User Management**: Secure user authentication and profile management.  
- ⏬ **Data Downloader**: Fetch historical stock data directly from Yahoo Finance.  

---

## 🛠 Tech Stack

- **Backend**: Python, Flask  
- **Data Processing**: Apache Spark 4.0.1  
- **Machine Learning**: PySpark MLlib, Scikit-learn  
- **Database**: MongoDB  
- **Frontend**: HTML, CSS, JavaScript  

---

## ⚙️ Setup and Installation Guide

Follow these steps to set up StockPulse on your machine.

### 1. Prerequisites

Make sure you have the following installed:

- **Python 3.11.0** → [Download](https://www.python.org/downloads/)  
- **OpenJDK 17** → required by Spark (we recommend [Eclipse Temurin 17](https://adoptium.net/temurin/releases/?version=17))  
- **Apache Spark 4.0.1** → [Download](https://spark.apache.org/downloads.html) (*pre-built for Hadoop*)  

---

### 2. Configure Environment Variables

#### JAVA_HOME  

**Windows (Command Prompt):**
```bash
setx JAVA_HOME "C:\Program Files\Eclipse Adoptium\jdk-17.0.x.x-hotspot"
setx PATH "%PATH%;%JAVA_HOME%\bin"
```

**macOS/Linux (`.zshrc` or `.bashrc`):**
```bash
export JAVA_HOME="/path/to/your/jdk-17"
export PATH=$JAVA_HOME/bin:$PATH
```

#### SPARK_HOME  

Unzip Spark into a folder without spaces (e.g., `C:\spark`).

**Windows:**
```bash
setx SPARK_HOME "C:\spark\spark-4.0.1-bin-hadoop3"
setx PATH "%PATH%;%SPARK_HOME%\bin"
```

**macOS/Linux (`.zshrc` or `.bashrc`):**
```bash
export SPARK_HOME="/path/to/your/spark-4.0.1-bin-hadoop3"
export PATH=$SPARK_HOME/bin:$PATH
```

⚠️ **Tip**: Restart your terminal/command prompt after setting these.

---

### 3. Project Setup

Clone the repository:
```bash
git clone <https://github.com/umair25591/StockPulse.git>
cd <your_project_directory>
```

Create and activate a virtual environment:
```bash

python -m venv venv


.\venv\Scripts\activate

```

Install dependencies:
```bash
pip install -r requirements.txt
```

Copy the example environment file:
```bash

copy .env.example .env

cp .env.example .env
```

Update `.env` with:
- Your **MongoDB connection string**
- A unique **secret key**
- **Given By Data Stromer**

---

## ▶️ Running the Application

1. Activate your **virtual environment**.  
2. Make sure **MongoDB** is running.  
3. Start the application:
   ```bash
   python app.py
   ```
4. Open your browser at:  
   👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

You should now see the **StockPulse Homepage**

---

## 👨‍💻 Author

Developed by **Data Stromer**
