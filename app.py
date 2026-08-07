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

# Global variables
current_data = None
PERSISTENT_FILE_PATH = os.path.join(app.config['UPLOAD_FOLDER'], 'persistent_data.csv')
trained_model = None
feature_columns = None
target_column = None
analytics_cache = None

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_persistent_data():
    global current_data, analytics_cache
    if os.path.exists(PERSISTENT_FILE_PATH):
        try:
            if PERSISTENT_FILE_PATH.endswith('.csv'):
                current_data = pd.read_csv(PERSISTENT_FILE_PATH)
            else:
                current_data = pd.read_excel(PERSISTENT_FILE_PATH)
            analytics_cache = None
            print(f"✅ Loaded persistent dataset: {current_data.shape}")
            return True
        except Exception as e:
            print(f"⚠️ Error loading persistent data: {e}")
            return False
    return False


# Load on startup
load_persistent_data()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dataset_loading')
def dataset_loading():
    return render_template('dataset_loading.html')


# -------------------------------------------------------------
# NEW ROUTE: Allows frontend to check if data is already loaded
# -------------------------------------------------------------
@app.route('/check_data_status', methods=['GET'])
def check_data_status():
    global current_data
    if current_data is None:
        load_persistent_data()  # Try to load if not in memory

    if current_data is not None:
        return jsonify({
            'loaded': True,
            'columns': current_data.columns.tolist(),
            'shape': current_data.shape
        })
    return jsonify({'loaded': False})


@app.route('/upload', methods=['POST'])
def upload_file():
    global current_data, analytics_cache

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

            current_data = full_df.copy()
            analytics_cache = None
            current_data.to_csv(PERSISTENT_FILE_PATH, index=False)

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
    global current_data, analytics_cache

    if current_data is None:
        if not load_persistent_data():
            return jsonify({'error': 'No data loaded. Please upload a dataset first.'}), 400

    action = request.json.get('action')
    result = {}

    try:
        if action == 'missing_values':
            method = request.json.get('method', 'drop')
            column = request.json.get('column')

            if column and column in current_data.columns:
                if method == 'drop':
                    current_data = current_data.dropna(subset=[column])
                elif method == 'mean':
                    current_data[column].fillna(current_data[column].mean(), inplace=True)
                elif method == 'median':
                    current_data[column].fillna(current_data[column].median(), inplace=True)
            else:
                if method == 'drop':
                    before = len(current_data)
                    current_data = current_data.dropna()
                    after = len(current_data)
                    result = {
                        'success': True,
                        'message': f'Dropped {before - after} rows with missing values',
                        'rows_removed': before - after
                    }
                else:
                    numeric_cols = current_data.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if method == 'mean':
                            current_data[col].fillna(current_data[col].mean(), inplace=True)
                        elif method == 'median':
                            current_data[col].fillna(current_data[col].median(), inplace=True)
                    result = {
                        'success': True,
                        'message': f'Missing values filled using {method}',
                        'null_counts': current_data.isnull().sum().to_dict()
                    }
            current_data.to_csv(PERSISTENT_FILE_PATH, index=False)
            analytics_cache = None

        elif action == 'encoding':
            column = request.json.get('column')
            if column and column in current_data.columns:
                le = LabelEncoder()
                current_data[column] = le.fit_transform(current_data[column].astype(str))
                result = {
                    'success': True,
                    'message': f'Column {column} encoded successfully'
                }
                current_data.to_csv(PERSISTENT_FILE_PATH, index=False)
                analytics_cache = None

        elif action == 'feature_scaling':
            columns = current_data.select_dtypes(include=[np.number]).columns.tolist()
            if columns:
                scaler = StandardScaler()
                current_data[columns] = scaler.fit_transform(current_data[columns])
                result = {
                    'success': True,
                    'message': 'Features scaled successfully',
                    'scaled_columns': columns
                }
                current_data.to_csv(PERSISTENT_FILE_PATH, index=False)
                analytics_cache = None

        elif action == 'outlier_detection':
            column = request.json.get('column')
            if column and column in current_data.columns:
                Q1 = current_data[column].quantile(0.25)
                Q3 = current_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = current_data[(current_data[column] < lower_bound) | (current_data[column] > upper_bound)]
                result = {
                    'success': True,
                    'message': f'Outliers detected in {column}',
                    'outliers_count': len(outliers)
                }

        elif action == 'remove_duplicates':
            before = len(current_data)
            current_data = current_data.drop_duplicates()
            after = len(current_data)
            result = {
                'success': True,
                'message': f'Removed {before - after} duplicate rows',
                'rows_removed': before - after,
                'remaining_rows': after
            }
            current_data.to_csv(PERSISTENT_FILE_PATH, index=False)
            analytics_cache = None

    except Exception as e:
        return jsonify({'error': str(e)}), 400

    return jsonify(result)


@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/train_model', methods=['POST'])
def train_model():
    global current_data, trained_model, feature_columns, target_column
    if current_data is None:
        if not load_persistent_data():
            return jsonify({'error': 'No data loaded'}), 400
    try:
        data = request.json
        target = data.get('target_column')
        model_type = data.get('model_type', 'regression')
        test_size = float(data.get('test_size', 0.2))

        if target not in current_data.columns:
            return jsonify({'error': f'Target column {target} not found'}), 400
        X = current_data.drop(columns=[target])
        y = current_data[target]
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


@app.route('/analytics')
def analytics():
    return render_template('analytics.html')


@app.route('/get_analytics', methods=['GET'])
def get_analytics():
    global current_data, analytics_cache
    if current_data is None:
        if not load_persistent_data():
            return jsonify({'error': 'No data loaded'}), 400

    if analytics_cache:
        return jsonify(analytics_cache)

    try:
        visualizations = []
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = current_data[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            img = BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
            img.seek(0)
            corr_plot = base64.b64encode(img.getvalue()).decode()
            visualizations.append({
                'title': 'Correlation Heatmap',
                'image': corr_plot,
                'type': 'correlation'
            })
            plt.close()

        if 'grade' in current_data.columns:
            plt.figure(figsize=(10, 6))
            grade_counts = current_data['grade'].value_counts()
            colors = ['#f7971e', '#ffd200', '#43e97b', '#4facfe', '#fa709a']
            bars = plt.bar(grade_counts.index, grade_counts.values, color=colors[:len(grade_counts)], edgecolor='white',
                           linewidth=2)
            plt.title('Student Grade Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Grade', fontsize=12)
            plt.ylabel('Number of Students', fontsize=12)
            plt.xticks(rotation=45)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}', ha='center', va='bottom',
                         fontweight='bold')
            plt.tight_layout()
            img = BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
            img.seek(0)
            bar_plot = base64.b64encode(img.getvalue()).decode()
            visualizations.append({
                'title': 'Grade Distribution (Bar Chart)',
                'image': bar_plot,
                'type': 'bar'
            })
            plt.close()

        if 'department' in current_data.columns:
            plt.figure(figsize=(10, 8))
            dept_counts = current_data['department'].value_counts()
            colors = ['#f7971e', '#ffd200', '#43e97b', '#4facfe', '#fa709a', '#a18cd1', '#fbc2eb']
            if len(dept_counts) > 6:
                top_depts = dept_counts[:6]
                other_count = dept_counts[6:].sum()
                top_depts['Others'] = other_count
                dept_counts = top_depts
            plt.pie(dept_counts.values, labels=dept_counts.index, autopct='%1.1f%%', colors=colors[:len(dept_counts)],
                    startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
            plt.title('Student Distribution by Department', fontsize=14, fontweight='bold')
            plt.tight_layout()
            img = BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
            img.seek(0)
            pie_plot = base64.b64encode(img.getvalue()).decode()
            visualizations.append({
                'title': 'Department Distribution (Pie Chart)',
                'image': pie_plot,
                'type': 'pie'
            })
            plt.close()

        placement_cols = [col for col in current_data.columns if
                          'placement' in col.lower() or 'placed' in col.lower() or 'status' in col.lower()]
        if placement_cols:
            placement_col = placement_cols[0]
            plt.figure(figsize=(10, 8))
            if current_data[placement_col].dtype == 'object':
                placement_counts = current_data[placement_col].value_counts()
                colors = ['#43e97b', '#fa709a', '#ffd200', '#4facfe']
                wedges, texts, autotexts = plt.pie(placement_counts.values, labels=placement_counts.index,
                                                   autopct='%1.1f%%', colors=colors[:len(placement_counts)],
                                                   startangle=90, wedgeprops={'width': 0.7},
                                                   textprops={'fontsize': 11, 'fontweight': 'bold'})
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                plt.title('Placement Status Overview', fontsize=14, fontweight='bold')
            else:
                placement_counts = current_data[placement_col].value_counts().sort_index()
                bars = plt.bar(placement_counts.index.astype(str), placement_counts.values,
                               color=['#43e97b', '#fa709a', '#ffd200'])
                plt.title('Placement Overview', fontsize=14, fontweight='bold')
                plt.xlabel(placement_col, fontsize=12)
                plt.ylabel('Count', fontsize=12)
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}', ha='center', va='bottom',
                             fontweight='bold')
            plt.tight_layout()
            img = BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
            img.seek(0)
            placement_plot = base64.b64encode(img.getvalue()).decode()
            visualizations.append({
                'title': 'Placement Overview',
                'image': placement_plot,
                'type': 'placement'
            })
            plt.close()

        for col in numeric_cols[:2]:
            if col not in ['score', 'grade']:
                plt.figure(figsize=(10, 6))
                sns.histplot(current_data[col], kde=True, color='#f7971e', bins=20)
                plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold')
                plt.xlabel(col, fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.tight_layout()
                img = BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
                img.seek(0)
                dist_plot = base64.b64encode(img.getvalue()).decode()
                visualizations.append({
                    'title': f'Distribution - {col}',
                    'image': dist_plot,
                    'type': 'distribution'
                })
                plt.close()

        stats = current_data.describe().to_html(classes='table table-striped')
        total_students = len(current_data)
        avg_score = float(current_data[numeric_cols].mean().mean()) if len(numeric_cols) > 0 else 0

        analytics_cache = {
            'success': True,
            'visualizations': visualizations,
            'stats': stats,
            'shape': current_data.shape,
            'total_students': total_students,
            'avg_score': avg_score
        }
        return jsonify(analytics_cache)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/help')
def help_page():
    return render_template('help.html')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)