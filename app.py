import os
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, render_template, redirect, url_for
import gseapy as gp

# -----------------------------
# Flask app setup
# -----------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------
# Load model, scaler, features
# -----------------------------
svm = joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))

# -----------------------------
# Routes
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('analyze', filename=file.filename))
    return render_template('index.html')


@app.route('/analyze/<filename>')
def analyze(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath, index_col=0)

    X = df[feature_names].copy()
    X_scaled = scaler.transform(X)

    y_pred = svm.predict(X_scaled)
    y_prob = svm.predict_proba(X_scaled)[:, 1]

    results = pd.DataFrame({
        'SampleID': df.index,
        'Prediction': ['Tumor' if p == 1 else 'Normal' for p in y_pred],
        'Probability': y_prob
    })

    results['sort_key'] = results['Prediction'].apply(lambda x: 0 if x == 'Tumor' else 1)
    results = results.sort_values(['sort_key', 'SampleID']).drop(columns='sort_key')

    tumor_samples = results[results['Prediction'] == 'Tumor']['SampleID'].tolist()

    return render_template(
        'results.html',
        results=results.to_dict(orient='records'),
        tumor_samples=tumor_samples,
        filename=filename
    )


@app.route('/sample/<filename>/<sample_id>')
def view_sample(filename, sample_id):
    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), index_col=0)
    X = df[feature_names].copy()

    if sample_id not in X.index:
        return f"Sample {sample_id} not found."

    sample_values = X.loc[sample_id]
    top_features = sample_values.sort_values(ascending=False).head(30)

    # Protein plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title(f"Top 30 Proteins in {sample_id}")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    protein_plot_url = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    # Pathway enrichment
    pathway_plot_url = None
    try:
        enr = gp.enrichr(
            gene_list=top_features.index.tolist(),
            gene_sets=['KEGG_2021_Human', 'Reactome_2022'],
            organism='Human',
            outdir=None
        )

        if not enr.results.empty:
            top_pathways = enr.results.sort_values('Combined Score', ascending=False).head(20)
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Combined Score', y='Term', data=top_pathways)
            plt.title(f"Top Pathways in {sample_id}")
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            pathway_plot_url = base64.b64encode(buf.getvalue()).decode()
            plt.close()
    except Exception as e:
        print("Enrichment error:", e)

    return render_template(
        'sample.html',
        sample_id=sample_id,
        protein_plot_url=protein_plot_url,
        pathway_plot_url=pathway_plot_url,
        filename=filename
    )
