from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from werkzeug.utils import secure_filename
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.secret_key = 'your_secret_key_here'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# PERSISTENT FILE PATH
PERSISTENT_FILE_PATH = os.path.join(app.config['UPLOAD_FOLDER'], 'persistent_data.csv')

# Global variables
trained_model = None
feature_columns = None
target_column = None
analytics_cache = None
eda_cache = None

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# CRITICAL FIX: SAFE DATA LOADER WITH AUTOMATIC CLEANUP
def get_data_from_disk():
    """Reads the file from disk and cleans it to prevent crashes."""
    if os.path.exists(PERSISTENT_FILE_PATH):
        try:
            if PERSISTENT_FILE_PATH.endswith('.csv'):
                df = pd.read_csv(PERSISTENT_FILE_PATH)
            else:
                df = pd.read_excel(PERSISTENT_FILE_PATH)

            # 1. Clean Infinite numbers
            df = df.replace([np.inf, -np.inf], np.nan)

            # 2. Drop completely empty rows
            df = df.dropna(how='all')

            print(f"📁 Loaded dataset successfully: {df.shape}")
            return df
        except Exception as e:
            print(f"⚠️ Error reading data: {e}")
            return None
    return None


@app.route('/check_data_status', methods=['GET'])
def check_data_status():
    df = get_data_from_disk()
    if df is not None:
        return jsonify({
            'loaded': True,
            'columns': df.columns.tolist(),
            'shape': df.shape
        })
    return jsonify({'loaded': False})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dataset_loading')
def dataset_loading():
    return render_template('dataset_loading.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global analytics_cache, eda_cache

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            if filename.endswith('.csv'):
                full_df = pd.read_csv(filepath)
                preview_df = pd.read_csv(filepath, nrows=100)
            else:
                full_df = pd.read_excel(filepath)
                preview_df = pd.read_excel(filepath, nrows=100)

            # Save permanently
            full_df.to_csv(PERSISTENT_FILE_PATH, index=False)
            analytics_cache = None
            eda_cache = None

            preview = preview_df.head(10).to_html(classes='table table-striped table-bordered')

            info = {
                'shape': f"{full_df.shape[0]} rows, {full_df.shape[1]} columns",
                'columns': full_df.columns.tolist(),
                'dtypes': full_df.dtypes.astype(str).to_dict(),
                'null_counts': full_df.isnull().sum().to_dict(),
                'preview': preview
            }

            return jsonify({
                'success': True,
                'message': f'File {filename} uploaded and saved permanently!',
                'info': info
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    return jsonify({'error': 'Invalid file format. Only CSV and Excel files are allowed.'}), 400


@app.route('/preprocessing')
def preprocessing():
    return render_template('preprocessing.html')


@app.route('/preprocess', methods=['POST'])
def preprocess():
    global analytics_cache, eda_cache

    df = get_data_from_disk()
    if df is None:
        return jsonify({'error': 'No data found on disk. Please upload a dataset first.'}), 400

    action = request.json.get('action')
    result = {}

    try:
        if action == 'missing_values':
            method = request.json.get('method', 'drop')
            column = request.json.get('column')

            if column and column in df.columns:
                if method == 'drop':
                    df = df.dropna(subset=[column])
                    result = {'success': True, 'message': f'Dropped rows with missing values in {column}'}
                elif method in ['mean', 'median']:
                    if pd.api.types.is_numeric_dtype(df[column]):
                        fill_value = df[column].mean() if method == 'mean' else df[column].median()
                        if pd.isna(fill_value):
                            result = {'success': False, 'error': f'Cannot calculate {method} on column "{column}".'}
                        else:
                            df[column].fillna(fill_value, inplace=True)
                            result = {'success': True, 'message': f'Missing values in "{column}" filled using {method}'}
                    else:
                        result = {'success': False, 'error': f'Cannot apply {method} to non-numeric column "{column}".'}
            else:
                if method == 'drop':
                    before = len(df)
                    df = df.dropna()
                    result = {'success': True, 'message': f'Dropped {before - len(df)} rows'}
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        fill_val = df[col].mean() if method == 'mean' else df[col].median()
                        if not pd.isna(fill_val):
                            df[col].fillna(fill_val, inplace=True)
                    result = {'success': True, 'message': f'Missing values filled using {method} on numeric columns'}

            df.to_csv(PERSISTENT_FILE_PATH, index=False)
            analytics_cache = None
            eda_cache = None

        elif action == 'encoding':
            column = request.json.get('column')
            if column and column in df.columns:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column].astype(str))
                result = {'success': True, 'message': f'Column {column} encoded successfully'}
                df.to_csv(PERSISTENT_FILE_PATH, index=False)
                analytics_cache = None
                eda_cache = None

        elif action == 'feature_scaling':
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if columns:
                scaler = StandardScaler()
                df[columns] = scaler.fit_transform(df[columns])
                result = {'success': True, 'message': 'Features scaled successfully', 'scaled_columns': columns}
                df.to_csv(PERSISTENT_FILE_PATH, index=False)
                analytics_cache = None
                eda_cache = None

        elif action == 'outlier_detection':
            column = request.json.get('column')
            if column and column in df.columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
                result = {'success': True, 'message': f'Outliers detected in {column}', 'outliers_count': len(outliers)}

        elif action == 'remove_duplicates':
            before = len(df)
            df = df.drop_duplicates()
            result = {'success': True, 'message': f'Removed {before - len(df)} duplicate rows'}
            df.to_csv(PERSISTENT_FILE_PATH, index=False)
            analytics_cache = None
            eda_cache = None

    except Exception as e:
        return jsonify({'error': str(e)}), 400

    return jsonify(result)


@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/train_model', methods=['POST'])
def train_model():
    global trained_model, feature_columns, target_column

    df = get_data_from_disk()
    if df is None:
        return jsonify({'error': 'No data found on disk. Please upload a dataset first.'}), 400

    try:
        data = request.json
        target = data.get('target_column')
        model_type = data.get('model_type', 'regression')
        test_size = float(data.get('test_size', 0.2))

        if target not in df.columns:
            return jsonify({'error': f'Target column {target} not found'}), 400
        X = df.drop(columns=[target])
        y = df[target]
        if model_type == 'regression' and not pd.api.types.is_numeric_dtype(y):
            return jsonify({'error': 'Target column must be numeric for regression'}), 400
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if model_type == 'regression':
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            metric = 'R2 Score'
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            metric = 'Accuracy'

        trained_model = model
        feature_columns = X.columns.tolist()
        target_column = target
        return jsonify({
            'success': True,
            'message': 'Model trained successfully!',
            'score': float(score),
            'metric': metric,
            'features': feature_columns,
            'target': target
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/make_prediction', methods=['POST'])
def make_prediction():
    global trained_model, feature_columns
    if trained_model is None:
        return jsonify({'error': 'No trained model found'}), 400
    try:
        data = request.json
        input_data = {}
        for feature in feature_columns:
            value = data.get(feature)
            if value is None:
                return jsonify({'error': f'Missing value for feature: {feature}'}), 400
            input_data[feature] = float(value)
        input_df = pd.DataFrame([input_data])
        prediction = trained_model.predict(input_df)
        return jsonify({
            'success': True,
            'prediction': float(prediction[0]),
            'message': 'Prediction made successfully!'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/students')
def students():
    return render_template('students.html')


@app.route('/get_students_data', methods=['GET'])
def get_students_data():
    df = get_data_from_disk()
    if df is None:
        return jsonify({'success': False, 'message': 'No dataset uploaded. Please go to Dataset Loading.'})

    try:
        preview_data = df.head(100).to_dict(orient='records')
        columns = df.columns.tolist()
        return jsonify({
            'success': True,
            'data': preview_data,
            'columns': columns,
            'total_rows': len(df)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/eda_info')
def eda_info():
    return render_template('eda_info.html')


@app.route('/get_eda_info', methods=['GET'])
def get_eda_info():
    global eda_cache
    if eda_cache:
        return jsonify(eda_cache)

    df = get_data_from_disk()
    if df is None:
        return jsonify({'error': 'No data found on disk. Please upload a dataset.'}), 400

    try:
        info = {
            'success': True,
            'rows': df.shape[0],
            'columns': df.shape[1],
            'column_names': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'null_total': int(df.isnull().sum().sum()),
            'memory_usage': f"{round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)} MB",
            'head': df.head(10).to_html(classes='table table-striped table-bordered table-sm', index=False),
            'describe': df.describe(include='all').to_html(classes='table table-striped table-bordered table-sm')
        }
        eda_cache = info
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/analytics')
def analytics():
    return render_template('analytics.html')


@app.route('/get_analytics', methods=['GET'])
def get_analytics():
    global analytics_cache
    if analytics_cache:
        return jsonify(analytics_cache)

    df = get_data_from_disk()
    if df is None:
        return jsonify({'error': 'No data found on disk. Please upload a dataset.'}), 400

    try:
        visualizations = []

        # CRITICAL FIX: Drop columns that are entirely empty before generating graphs
        df = df.dropna(axis=1, how='all')

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=['object']).columns

        # 1. PIE CHART (Categorical)
        if len(cat_cols) > 0:
            col = cat_cols[0]
            plt.figure(figsize=(8, 6))
            counts = df[col].value_counts()
            if len(counts) > 6:
                top_counts = counts[:6]
                top_counts['Others'] = counts[6:].sum()
                counts = top_counts
            colors = ['#f7971e', '#ffd200', '#43e97b', '#4facfe', '#fa709a', '#a18cd1']
            plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors[:len(counts)], startangle=90,
                    textprops={'fontsize': 11})
            plt.title(f'Distribution: {col}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            img = BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
            img.seek(0)
            visualizations.append({
                'title': f'Pie Chart: {col}',
                'image': base64.b64encode(img.getvalue()).decode(),
                'type': 'pie'
            })
            plt.close()

        # 2. BAR GRAPH
        if len(cat_cols) > 1:
            col = cat_cols[1]
            plt.figure(figsize=(10, 6))
            counts = df[col].value_counts()
            if len(counts) > 10:
                counts = counts[:10]
            sns.barplot(x=counts.index, y=counts.values, palette='viridis')
            plt.title(f'Bar Chart: {col}', fontsize=14, fontweight='bold')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            img = BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
            img.seek(0)
            visualizations.append({
                'title': f'Bar Chart: {col}',
                'image': base64.b64encode(img.getvalue()).decode(),
                'type': 'bar'
            })
            plt.close()
        elif len(numeric_cols) > 0:
            col = numeric_cols[0]
            plt.figure(figsize=(10, 6))
            counts = df[col].value_counts().sort_index()
            if len(counts) > 10:
                counts = counts[:10]
            sns.barplot(x=counts.index.astype(str), y=counts.values, palette='magma')
            plt.title(f'Bar Chart (Numeric): {col}', fontsize=14, fontweight='bold')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            img = BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
            img.seek(0)
            visualizations.append({
                'title': f'Bar Chart: {col}',
                'image': base64.b64encode(img.getvalue()).decode(),
                'type': 'bar'
            })
            plt.close()

        # 3. Correlation Heatmap
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
            plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            img = BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
            img.seek(0)
            visualizations.append({
                'title': 'Correlation Heatmap',
                'image': base64.b64encode(img.getvalue()).decode(),
                'type': 'heatmap'
            })
            plt.close()

        # 4. Distribution (Histogram)
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True, color='#f7971e', bins=30)
            plt.title(f'Distribution: {col}', fontsize=14, fontweight='bold')
            plt.xlabel(col)
            plt.tight_layout()
            img = BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
            img.seek(0)
            visualizations.append({
                'title': f'Distribution: {col}',
                'image': base64.b64encode(img.getvalue()).decode(),
                'type': 'distribution'
            })
            plt.close()

        # 5. Boxplot (Outliers)
        if len(numeric_cols) > 1:
            col = numeric_cols[1]
            plt.figure(figsize=(8, 6))
            sns.boxplot(y=df[col], color='#43e97b')
            plt.title(f'Outliers: {col}', fontsize=14, fontweight='bold')
            plt.ylabel(col)
            plt.tight_layout()
            img = BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
            img.seek(0)
            visualizations.append({
                'title': f'Outliers: {col}',
                'image': base64.b64encode(img.getvalue()).decode(),
                'type': 'boxplot'
            })
            plt.close()

        analytics_cache = {
            'success': True,
            'visualizations': visualizations
        }
        return jsonify(analytics_cache)
    except Exception as e:
        # This catches ANY crash and sends the REAL error to your browser console/alert
        return jsonify({'error': f"Python Crash: {str(e)}"}), 400


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/help')
def help_page():
    return render_template('help.html')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)