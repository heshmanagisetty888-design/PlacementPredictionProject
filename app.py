import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global DataFrame
GLOBAL_DF = None
SAVED_FILE_PATH = os.path.join(app.config['UPLOAD_FOLDER'], 'persistent_dataset.csv')


def load_persistent_data():
    """Loads data from disk if it exists, so it survives a server restart/refresh"""
    global GLOBAL_DF
    if os.path.exists(SAVED_FILE_PATH):
        try:
            GLOBAL_DF = pd.read_csv(SAVED_FILE_PATH)
            # Clean data
            GLOBAL_DF = GLOBAL_DF.replace([np.inf, -np.inf], np.nan).fillna(0)
            print(f"Loaded persistent dataset: {GLOBAL_DF.shape}")
            return True
        except Exception as e:
            print(f"Error loading persistent data: {e}")
            return False
    return False


# Auto-load on startup
load_persistent_data()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dataset_loading')
def dataset_loading():
    return render_template('index.html')  # Sending user to the same dashboard


@app.route('/upload_data', methods=['POST'])
def upload_data():
    global GLOBAL_DF
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            if filename.endswith('.csv'):
                GLOBAL_DF = pd.read_csv(filepath)
            elif filename.endswith(('.xls', '.xlsx')):
                GLOBAL_DF = pd.read_excel(filepath)
            else:
                return jsonify({'success': False, 'message': 'Unsupported file type'})

            # Clean up
            GLOBAL_DF = GLOBAL_DF.replace([np.inf, -np.inf], np.nan).fillna(0)

            # SAVE A COPY TO DISK SO IT PERSISTS ON REFRESH
            GLOBAL_DF.to_csv(SAVED_FILE_PATH, index=False)

            return jsonify({
                'success': True,
                'message': f'File uploaded successfully. Shape: {GLOBAL_DF.shape}'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})


@app.route('/get_dashboard_stats', methods=['GET'])
def get_dashboard_stats():
    global GLOBAL_DF

    # If data is None, try loading from disk one more time
    if GLOBAL_DF is None:
        if not load_persistent_data():
            return jsonify({'success': False, 'message': 'No dataset loaded yet'})

    try:
        total_students = len(GLOBAL_DF)
        total_features = len(GLOBAL_DF.columns)

        # Calculate Avg Score & Pass rate
        numeric_cols = GLOBAL_DF.select_dtypes(include=[np.number]).columns
        avg_score = 0
        pass_rate = 0

        if len(numeric_cols) > 0:
            main_col = numeric_cols[0]
            avg_score = float(GLOBAL_DF[main_col].mean())
            threshold = GLOBAL_DF[main_col].quantile(0.6)  # Top 60% considered "pass"
            pass_rate = (len(GLOBAL_DF[GLOBAL_DF[main_col] >= threshold]) / total_students) * 100

        return jsonify({
            'success': True,
            'total_students': total_students,
            'avg_score': round(avg_score, 1),
            'pass_rate': round(pass_rate, 1),
            'total_features': total_features
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/get_analytics', methods=['GET'])
def get_analytics():
    global GLOBAL_DF
    if GLOBAL_DF is None:
        if not load_persistent_data():
            return jsonify({'success': False, 'message': 'No data loaded.'})

    try:
        numeric_cols = GLOBAL_DF.select_dtypes(include=[np.number]).columns
        cat_cols = GLOBAL_DF.select_dtypes(include=['object']).columns

        # Data for pie chart (Use 1st categorical column)
        pie_labels = []
        pie_data = []
        if len(cat_cols) > 0:
            dist_col = cat_cols[0]
            counts = GLOBAL_DF[dist_col].value_counts()
            pie_labels = list(counts.index)
            pie_data = list(counts.values)

        # Data for distribution chart (Use 1st numeric column)
        dist_values = []
        dist_col_name = "No Numeric Data"
        if len(numeric_cols) > 0:
            dist_col = numeric_cols[0]
            dist_col_name = dist_col
            dist_values = GLOBAL_DF[dist_col].dropna().tolist()[:500]  # Limit for performance

        return jsonify({
            'success': True,
            'total_students': len(GLOBAL_DF),
            'avg_score': round(float(GLOBAL_DF[numeric_cols[0]].mean()), 1) if len(numeric_cols) > 0 else 0,
            'total_features': len(GLOBAL_DF.columns),
            'pie_labels': pie_labels,
            'pie_data': pie_data,
            'dist_values': dist_values,
            'dist_col_name': dist_col_name
        })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


if __name__ == '__main__':
    app.run(debug=True)