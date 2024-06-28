import base64
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import io
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
import networkx as nx
from joblib import dump
from joblib import load
import squarify

app = Flask(__name__)

UPLOAD_FOLDER = './'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('indexnew.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, 'uploadedfile.xlsx')
        file.save(file_path)

        # Read the Excel file
        data = pd.read_excel(file_path)

        plots_base64 = []
        sns.countplot(y='Source of Income', data=data, order=data['Source of Income'].value_counts().index)
        plt.title('Distribution of Occupation')
        plt.xlabel('Count')
        plt.ylabel('Occupation')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plots_base64.append(img_base64)
        plt.clf()  # Clear the current figure


        data['YearOfBirth'] = pd.to_datetime(data['DateofBirth']).dt.year
        sns.histplot(data['YearOfBirth'], bins=30, kde=True)
        plt.title('Distribution of Year of Birth')
        plt.xlabel('Year of Birth')
        plt.ylabel('Count')

        img1 = io.BytesIO()
        plt.savefig(img1, format='png')
        img1.seek(0)
        img_base64 = base64.b64encode(img1.getvalue()).decode('utf-8')
        plots_base64.append(img_base64)
        
        return jsonify(plots_base64)
    
def prepare_data_summary(data):
    # Convert the first few rows of the DataFrame to JSON
    data_head = data.head().to_json(orient='split')
    
    # Calculate missing value summary
    missing_values = data.isnull().sum().to_json()

    # Get data types
    data_types = data.dtypes.astype(str).to_json()

    # Get descriptive statistics
    describe = data.describe().to_json(orient='split')

    return jsonify({
        'data_head': data_head,
        'missing_values': missing_values,
        'data_types': data_types,
        'describe': describe
    })

@app.route('/dataPreview', methods=['GET'])
def data_preview():
    data = pd.read_excel('uploadedfile.xlsx')
    return prepare_data_summary(data)

@app.route('/drop-missing-columns', methods=['POST'])
def drop_missing_columns():
    data = pd.read_excel('uploadedfile.xlsx')
    # Drop columns where all values are missing except the header
    data.dropna(axis=1, how='all', inplace=True)

    return prepare_data_summary(data)

@app.route('/encode_data', methods=['POST'])
def encode_data():
    data = pd.read_excel('uploadedfile.xlsx')
    data.dropna(axis=1, how='all', inplace=True)
    if data.empty:
        return "No data available. Please upload a file first.", 400

    # Separate numerical and non-numerical columns
    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    categorical_data = data.select_dtypes(include=['object'])

    # Handle missing values using KNN Imputer for numerical data
    imputer = KNNImputer(n_neighbors=5)
    numerical_data_imputed = imputer.fit_transform(numerical_data)

    # Encode non-numerical data
    label_encoders = {}
    for column in categorical_data.columns:
        le = LabelEncoder()
        categorical_data[column] = le.fit_transform(categorical_data[column].astype(str))
        label_encoders[column] = le

    # Combine imputed numerical data with encoded categorical data
    numerical_data_imputed_df = pd.DataFrame(numerical_data_imputed, columns=numerical_data.columns)
    data_processed = pd.concat([numerical_data_imputed_df, categorical_data.reset_index(drop=True)], axis=1)

    return prepare_data_summary(data_processed)

@app.route('/parse-date-and-calculate-age', methods=['POST'])
def parse_date_and_calculate_age():
    data = pd.read_excel('uploadedfile.xlsx')
    data.dropna(axis=1, how='all', inplace=True)
    if data.empty:
        return "No data available. Please upload a file first.", 400

    date_column = 'DateofBirth'
    if date_column not in data.columns:
        return f"Column '{date_column}' not found in data.", 400

    # Parse date column and calculate age
    current_year = datetime.now().year
    data['DateofBirth'] = pd.to_datetime(data[date_column], errors='coerce').dt.year
    data['Age'] = current_year - data['DateofBirth']
    data.drop(columns=['DateofBirth'], inplace=True)
    
    return prepare_data_summary(data)

@app.route('/detect-outliers', methods=['POST'])
def detect_outliers():
    data = pd.read_excel('uploadedfile.xlsx')
    data.dropna(axis=1, how='all', inplace=True)
    if data.empty:
        return "No data available. Please upload a file first.", 400

    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    categorical_data = data.select_dtypes(include=['object'])

    # Handle missing values using KNN Imputer for numerical data
    imputer = KNNImputer(n_neighbors=5)
    numerical_data_imputed = imputer.fit_transform(numerical_data)

    # Encode non-numerical data
    label_encoders = {}
    for column in categorical_data.columns:
        le = LabelEncoder()
        categorical_data[column] = le.fit_transform(categorical_data[column].astype(str))
        label_encoders[column] = le

    # Combine imputed numerical data with encoded categorical data
    numerical_data_imputed_df = pd.DataFrame(numerical_data_imputed, columns=numerical_data.columns)
    data_processed = pd.concat([numerical_data_imputed_df, categorical_data.reset_index(drop=True)], axis=1)
    z_scores = np.abs((data_processed - data_processed.mean()) / data_processed.std())
    outliers = z_scores > 3  # Z-score threshold
    outlier_counts = outliers.sum()
    outliers_filtered = outlier_counts[outlier_counts > 0]  # Filter out columns with no outliers
    outlier_summary = outliers_filtered.to_json()


    return jsonify({'outliers': outlier_summary})

@app.route('/handle-outliers', methods=['POST'])
def handle_outliers():
    data = pd.read_excel('risksummaryfactor2022.xlsx')
    data.dropna(axis=1, how='all', inplace=True)
    if data.empty:
        return "No data available. Please upload a file first.", 400
    
    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    categorical_data = data.select_dtypes(include=['object'])

    # Handle missing values using KNN Imputer for numerical data
    imputer = KNNImputer(n_neighbors=5)
    numerical_data_imputed = imputer.fit_transform(numerical_data)

    # Encode non-numerical data
    label_encoders = {}
    for column in categorical_data.columns:
        le = LabelEncoder()
        categorical_data[column] = le.fit_transform(categorical_data[column].astype(str))
        label_encoders[column] = le

    # Combine imputed numerical data with encoded categorical data
    numerical_data_imputed_df = pd.DataFrame(numerical_data_imputed, columns=numerical_data.columns)
    data_processed = pd.concat([numerical_data_imputed_df, categorical_data.reset_index(drop=True)], axis=1)

    z_scores = np.abs((data_processed - data_processed.mean()) / data_processed.std())
    outliers = z_scores > 3  # Z-score threshold
    data_cleaned = data_processed.copy()
    for column in data_processed.columns:
        data_cleaned[column][outliers[column]] = data_processed[column].mean()
    
    # Save the processed data back to an Excel file
    data_cleaned.to_excel('uploadedfile.xlsx', index=False)
    dump(label_encoders, 'label_encoders.joblib')
    # return prepare_data_summary(data_cleaned)

@app.route('/create-features', methods=['POST'])
def create_features():
    data = pd.read_excel('uploadedfile.xlsx')
    data.dropna(axis=1, how='all', inplace=True)
    if data.empty:
        return "No data available. Please upload a file first.", 400

    # Polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    imputer = KNNImputer(n_neighbors=5)
    numerical_data_imputed = imputer.fit_transform(numerical_data)
    # Combine imputed numerical data with encoded categorical data
    numerical_data_imputed_df = pd.DataFrame(numerical_data_imputed, columns=numerical_data.columns)
    poly_features = poly.fit_transform(numerical_data_imputed_df)
    poly_feature_names = poly.get_feature_names_out(numerical_data_imputed_df.columns)
    poly_features_df = pd.DataFrame(poly_features, columns=[f'poly_{name}' for name in poly_feature_names])

    # Combine original data with polynomial features
    data_enhanced = pd.concat([data.reset_index(drop=True), poly_features_df.reset_index(drop=True)], axis=1)

    return prepare_data_summary(data_enhanced)

@app.route('/dimensionality-reduction', methods=['POST'])
def dimensionality_reduction():
    data = pd.read_excel('uploadedfile.xlsx')
    data.dropna(axis=1, how='all', inplace=True)
    if data.empty:
        return "No data available. Please upload a file first.", 400

    # Perform PCA
    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    imputer = KNNImputer(n_neighbors=5)
    numerical_data_imputed = imputer.fit_transform(numerical_data)
    # Combine imputed numerical data with encoded categorical data
    numerical_data_imputed_df = pd.DataFrame(numerical_data_imputed, columns=numerical_data.columns)
    pca = PCA(n_components=2)  # Reducing to 2 dimensions for simplicity
    pca_result = pca.fit_transform(numerical_data_imputed_df)
    pca_df = pd.DataFrame(data=pca_result, columns=['Principal Component 1', 'Principal Component 2'])

    # Variance explained
    explained_variance = pca.explained_variance_ratio_.tolist()

    return jsonify({
        'original_data': data.head().to_json(orient='split'),
        'pca_data': pca_df.head().to_json(orient='split'),
        'explained_variance': explained_variance
    })
    
    data_enhanced = pd.concat([data.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)
    
    return prepare_data_summary(data_enhanced)

@app.route('/perform-clustering', methods=['POST'])
def perform_clustering():
    data = pd.read_excel('uploadedfile.xlsx')
    data.dropna(axis=1, how='all', inplace=True)
    if data.empty:
        return "No data available. Please upload a file first.", 400
    
    # Perform PCA
    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    imputer = KNNImputer(n_neighbors=5)
    numerical_data_imputed = imputer.fit_transform(numerical_data)
    # Combine imputed numerical data with encoded categorical data
    numerical_data_imputed_df = pd.DataFrame(numerical_data_imputed, columns=numerical_data.columns)
    pca = PCA(n_components=2)  # Reducing to 2 dimensions for simplicity
    pca_result = pca.fit_transform(numerical_data_imputed_df)
    data_reduced_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data_reduced_df)
    data_reduced_df['Cluster'] = clusters

    # Evaluate clustering
    sil_score = silhouette_score(data_reduced_df[['PC1', 'PC2']], clusters)
    print(f'Silhouette Score: {sil_score}')

    plots_base64 = []
    plt.figure(figsize=(10, 6))
    plt.scatter(data_reduced_df['PC1'], data_reduced_df['PC2'], c=clusters, cmap='viridis')
    plt.title('Customer Segmentation')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    # plt.savefig('static/customer_segmentation.png')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plots_base64.append(img_base64)
    plt.clf()  # Clear the current figure


    # Fit the model for anomaly detection
    iso_forest = IsolationForest(contamination=0.1)
    numerical_data_imputed_df['Anomaly'] = iso_forest.fit_predict(numerical_data_imputed_df)

    # Visualize the anomalies
    plt.figure(figsize=(10, 6))
    plt.scatter(data_reduced_df['PC1'], data_reduced_df['PC2'], c=numerical_data_imputed_df['Anomaly'], cmap='coolwarm')
    plt.title('Anomaly Detection')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Anomaly')

    img1 = io.BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    img_base64 = base64.b64encode(img1.getvalue()).decode('utf-8')
    plots_base64.append(img_base64)
        
    # return send_file(img, mimetype='image/png')
    return jsonify(plots_base64)
    # return sil_score

@app.route('/visualization', methods=['POST'])
def visualization():
    data = pd.read_excel('uploadedfile.xlsx')
    data.dropna(axis=1, how='all', inplace=True)
    if data.empty:
        return "No data available. Please upload a file first.", 400
    
    # Perform PCA
    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    imputer = KNNImputer(n_neighbors=5)
    numerical_data_imputed = imputer.fit_transform(numerical_data)
    # Combine imputed numerical data with encoded categorical data
    numerical_data_imputed_df = pd.DataFrame(numerical_data_imputed, columns=numerical_data.columns)
    pca = PCA(n_components=2)  # Reducing to 2 dimensions for simplicity
    pca_result = pca.fit_transform(numerical_data_imputed_df)
    data_reduced_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data_reduced_df)
    data_reduced_df['Cluster'] = clusters

    plots_base64 = []
    plt.figure(figsize=(10, 6))
    for cluster in set(clusters):
        cluster_data = data_reduced_df[data_reduced_df['Cluster'] == cluster]
        plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}')
    
    # Plot cluster centers
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='X', label='Centers')
    
    plt.title('Customer Segmentation with Nodes and Centers')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plots_base64.append(img_base64)
    plt.clf()  # Clear the current figure
    return jsonify(plots_base64)

@app.route('/process/<filename>')
def process_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Implement your data processing here
    return f"File {filename} uploaded and ready for processing."

# def visualize_clusters(data, kmeans_model):
#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=kmeans_model.labels_, cmap='viridis')
#     centers = kmeans_model.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='X', label='Centers')
#     plt.title('Customer Segmentation with K-means Clustering')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.legend()
#     plt.colorbar(scatter)
#     plt.show()

# data = pd.read_excel('risksummary2022.xlsx')
# data.dropna(axis=1, how='all', inplace=True)
# print("drop na")
# numerical_data = data.select_dtypes(include=['float64', 'int64'])
# categorical_data = data.select_dtypes(include=['object'])

# # Handle missing values using KNN Imputer for numerical data
# imputer = KNNImputer(n_neighbors=5)
# numerical_data_imputed = imputer.fit_transform(numerical_data)

# # Encode non-numerical data
# label_encoders = {}
# for column in categorical_data.columns:
#     le = LabelEncoder()
#     categorical_data[column] = le.fit_transform(categorical_data[column].astype(str))
#     label_encoders[column] = le

# # Combine imputed numerical data with encoded categorical data
# numerical_data_imputed_df = pd.DataFrame(numerical_data_imputed, columns=numerical_data.columns)
# data_processed = pd.concat([numerical_data_imputed_df, categorical_data.reset_index(drop=True)], axis=1)
# print("imputed")
# z_scores = np.abs((data_processed - data_processed.mean()) / data_processed.std())
# outliers = z_scores > 3  # Z-score threshold
# data_cleaned = data_processed.copy()
# for column in data_processed.columns:
#     data_cleaned[column][outliers[column]] = data_processed[column].mean()
# print("cleaned")
# # Save the processed data back to an Excel file
# # data_cleaned.to_excel('uploadedfile.xlsx', index=False)
# # Perform PCA
# numerical_data = data_cleaned.select_dtypes(include=['float64', 'int64'])
# imputer = KNNImputer(n_neighbors=5)
# numerical_data_imputed = imputer.fit_transform(numerical_data)
# # Combine imputed numerical data with encoded categorical data
# numerical_data_imputed_df = pd.DataFrame(numerical_data_imputed, columns=numerical_data.columns)

# data_processed = pd.concat([numerical_data_imputed_df, categorical_data.reset_index(drop=True)], axis=1)

# # Perform K-means clustering on the processed data
# kmeans = KMeans(n_clusters=3)
# clusters = kmeans.fit_predict(data_processed)
# data_processed['Cluster'] = clusters

# # Evaluate clustering
# sil_score = silhouette_score(data_processed.drop(columns=['Cluster']), clusters)
# print(f'Silhouette Score: {sil_score}')

# # Save the processed data back to an Excel file
# data_processed.to_excel('uploadedfile.xlsx', index=False)
# # pca = PCA(n_components=2)  # Reducing to 2 dimensions for simplicity
# # pca_result = pca.fit_transform(numerical_data_imputed_df)
# # data_reduced_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

# # # Perform K-means clustering
# # kmeans = KMeans(n_clusters=3)
# # clusters = kmeans.fit_predict(data_reduced_df)
# # data_reduced_df['Cluster'] = clusters

# # plots_base64 = []
# # plt.figure(figsize=(10, 6))
# # for cluster in set(clusters):
# #     cluster_data = data_reduced_df[data_reduced_df['Cluster'] == cluster]
# #     plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}')

# # # Plot cluster centers
# # centers = kmeans.cluster_centers_
# # plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='X', label='Centers')

# # plt.title('Customer Segmentation with Nodes and Centers')
# # plt.xlabel('Principal Component 1')
# # plt.ylabel('Principal Component 2')
# # plt.legend()
# # plt.savefig('customer_segmentation.png')
# visualize_clusters(data_processed.drop(columns=['Cluster']), KMeans(n_clusters=3).fit(data_processed.drop(columns=['Cluster'])))

def plot_heatmap(data_processed):
    cluster_means = data_processed.groupby('Cluster').mean()
    sns.heatmap(cluster_means, annot=True, cmap='viridis')
    plt.title('Cluster Feature Means')
    plt.show()

def display_cluster_profiles(data_processed):
    cluster_summary = data_processed.groupby('Cluster').agg(['mean', 'std', 'min', 'max'])
    print(cluster_summary)

def visualize_clusters(data_processed, kmeans_model, cluster_labels):
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data_processed.drop(columns=['Cluster', 'ClusterLabel']))
    data_reduced_df = pd.DataFrame(data_reduced, columns=['PC1', 'PC2'])
    data_reduced_df['Cluster'] = data_processed['Cluster']
    data_reduced_df['ClusterLabel'] = data_reduced_df['Cluster'].map(cluster_labels)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_reduced_df['PC1'], data_reduced_df['PC2'], c=data_reduced_df['Cluster'], cmap='viridis')
    centers = kmeans_model.cluster_centers_
    centers_reduced = PCA(n_components=2).fit_transform(centers)
    plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], s=300, c='red', marker='X', label='Centers')
    plt.title('Customer Segmentation with K-means Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.colorbar(scatter, label='Cluster')
    plt.show()
    
def predict_new_customer_cluster(new_customer_data, kmeans_model, label_encoders, imputer, pca, cluster_labels):
    numerical_data = new_customer_data.select_dtypes(include=['float64', 'int64'])
    categorical_data = new_customer_data.select_dtypes(include=['object'])

    # Impute missing values
    numerical_data_imputed = imputer.transform(numerical_data)

    # Encode categorical data
    for column in categorical_data.columns:
        le = label_encoders[column]
        categorical_data[column] = le.transform(categorical_data[column].astype(str))

    # Combine numerical and categorical data
    numerical_data_imputed_df = pd.DataFrame(numerical_data_imputed, columns=numerical_data.columns)
    new_customer_processed = pd.concat([numerical_data_imputed_df, categorical_data.reset_index(drop=True)], axis=1)

    # Standardize the data
    # new_customer_scaled = scaler.transform(new_customer_processed)

    # Apply PCA
    new_customer_reduced = pca.transform(new_customer_processed)

    # Predict the cluster
    cluster = kmeans_model.predict(new_customer_reduced)
    cluster_label = cluster_labels[cluster[0]]

    return cluster[0], cluster_label

def perform_clustering1(file_path, output_path):
    data = pd.read_excel(file_path)
    data.dropna(axis=1, how='all', inplace=True)
    if data.empty:
        return "No data available. Please upload a file first.", 400
    
    print('Data loaded and cleaned.')
    
    # binary_features = ['Is PEP', 'High Net Worth House-wife']
    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    # numerical_features = [col for col in data.select_dtypes(include=['float64', 'int64']).columns if col not in binary_features]
    categorical_data = data.select_dtypes(include=['object'])

    # Ensure Source of Income is treated as string
    if 'Source of Income' in categorical_data:
        data['Source of Income'] = data['Source of Income'].astype(str)
    
    # Separate binary features and fill NaN values with the mode
    # binary_data = data[binary_features].fillna(data[binary_features].mode().iloc[0])


    # Handle missing values using KNN Imputer for numerical data
    # numerical_data = data[numerical_features]
    imputer = KNNImputer(n_neighbors=5)
    numerical_data_imputed = imputer.fit_transform(numerical_data)

    # Encode non-numerical data
    label_encoders = {}
    for column in categorical_data.columns:
        le = LabelEncoder()
        categorical_data[column] = le.fit_transform(categorical_data[column].astype(str))
        label_encoders[column] = le

    # Combine imputed numerical data with encoded categorical data
    numerical_data_imputed_df = pd.DataFrame(numerical_data_imputed, columns=numerical_data.columns)
    data_processed = pd.concat([numerical_data_imputed_df, categorical_data.reset_index(drop=True)], axis=1)
    
    print('Data imputation and encoding complete.')

    # Apply PCA
    pca = PCA(n_components=2) 
    # pca = PCA(n_components=0.95, random_state=42)  # Retain 95% of variance
    data_reduced = pca.fit_transform(data_processed)
    print('PCA applied.')

    # Perform K-means clustering on the reduced data
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data_reduced)
    data_processed['Cluster'] = clusters
    print('Clustering complete.')

    # Analyze cluster characteristics
    risk_features = ['Expected Transaction Volume', 'Is PEP', 'High Net Worth House-wife']  # Example risk-related features
    cluster_means = data_processed.groupby('Cluster')[risk_features].mean()

    # Determine which cluster is high risk, medium risk, and low risk
    risk_scores = cluster_means.mean(axis=1)
    cluster_labels = {}
    cluster_labels[risk_scores.idxmax()] = 'High Risk'
    cluster_labels[risk_scores.idxmin()] = 'Low Risk'
    remaining_cluster = set(cluster_means.index) - set([risk_scores.idxmax(), risk_scores.idxmin()])
    cluster_labels[remaining_cluster.pop()] = 'Medium Risk'

    data_processed['ClusterLabel'] = data_processed['Cluster'].map(cluster_labels)
    print('Cluster labeling complete.')

    # Evaluate clustering
    # sil_score = silhouette_score(data_reduced, clusters)
    # dbi_score = davies_bouldin_score(data_reduced, clusters)
    # print(f'Silhouette Score: {sil_score}')
    # print(f'Davies-Bouldin Index: {dbi_score}')

    # Save the processed data back to an Excel file
    # data_processed.to_excel(output_path, index=False)
    # print('Processed data saved.')

    # Save clustering results and model
    results_file = 'clustering_results.xlsx'
    model_file = 'kmeans_model.joblib'
    data_processed.to_excel(results_file, index=False)
    dump(kmeans, model_file)
    dump(imputer, 'imputer.joblib')
    dump(label_encoders, 'label_encoders.joblib')
    dump(cluster_labels, 'cluster_labels.joblib')
    dump(pca, 'pca_model.joblib')

    return kmeans, imputer, label_encoders, cluster_labels
    # sil_score, dbi_score


@app.route('/predict', methods=['POST'])
def predict():
    #Example usage
    # file_path = 'risksummarytest.xlsx'
    # output_path = 'processedfile.xlsx'
    # kmeans_model, label_encoders, imputer, cluster_labels, sil_score, dbi_score = perform_clustering1(file_path, output_path)
    results_file = 'clustering_results.xlsx'
    model_file = 'kmeans_model.joblib'
    # data_processed = pd.read_excel(results_file)
    kmeans_model = load(model_file)
    imputer = load('imputer.joblib')
    label_encoders = load('label_encoders.joblib')
    cluster_labels = load('cluster_labels.joblib')
    pca = load('pca_model.joblib')
    # print(data_summary)
    # print(f'Silhouette Score: {sil_score}')
    # print(f'Davies-Bouldin Index: {dbi_score}')

    json_data = request.get_json()
    new_customer_data = pd.DataFrame(json_data)

    # Predict the cluster for the new customer
    cluster, cluster_label = predict_new_customer_cluster(new_customer_data, kmeans_model, label_encoders, imputer, pca, cluster_labels)
    return jsonify({
        'message': f'The new customer belongs to cluster: {cluster} ({cluster_label})'
    })

# handle_outliers()
file_path = 'uploadedfile.xlsx'
output_path = 'processedfile.xlsx'
kmeans_model, imputer, label_encoders, cluster_labels = perform_clustering1(file_path, output_path)
# , sil_score, dbi_score
# print(f'Silhouette Score: {sil_score}')
# print(f'Davies-Bouldin Index: {dbi_score}')

@app.route('/cluster_analysis', methods=['POST'])
def cluster_analysis():
    # Load the saved models and data
    cluster_labels = load('cluster_labels.joblib')
    label_encoders = load('label_encoders.joblib')
    data_processed = pd.read_excel('clustering_results.xlsx')

    # Decode categorical values
    for column in label_encoders:
        le = label_encoders[column]
        data_processed[column] = le.inverse_transform(data_processed[column])

    # # Check the dataframe to ensure categorical values are decoded
    # print(data_processed.head())

    # # Plot cluster distribution
    # plt.figure(figsize=(10, 6))
    # sns.countplot(x='Cluster', data=data_processed, palette='viridis')
    # plt.title('Cluster Distribution')
    # plt.xlabel('Cluster')
    # plt.ylabel('Count')
    # plt.show()

    plots_base64 = []

    # List of numerical features to plot
    numerical_features = ['Expected Transaction Volume']  # Replace with your numerical features

    # Create bar graphs for numerical features
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='ClusterLabel', y=feature, data=data_processed, palette='viridis')
        plt.title(f'Cluster vs {feature}')
        plt.xlabel('Cluster')
        plt.ylabel(feature)
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plots_base64.append({'img_base64': img_base64, 'label': ''})
        plt.clf()  # Clear the current figure

    # Create line graphs for numerical features
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='ClusterLabel', y=feature, data=data_processed, marker='o')
        plt.title(f'Cluster vs {feature}')
        plt.xlabel('Cluster')
        plt.ylabel(feature)
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plots_base64.append({'img_base64': img_base64, 'label': ''})
        plt.clf()  # Clear the current figure

    # Create bar plots for binary features with values 0 and 1
    binary_features = ['Is PEP', 'High Net Worth House-wife']
    for feature in binary_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, hue='ClusterLabel', data=data_processed[data_processed[feature].isin([0, 1])], palette='viridis')
        plt.title(f'{feature} Distribution by Cluster')
        plt.xlabel(feature)
        plt.ylabel('Count')
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plots_base64.append({'img_base64': img_base64, 'label': ''})
        plt.clf()  # Clear the current figure

    # Create bar plots for gender
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Gender', hue='ClusterLabel', data=data_processed, palette='viridis')
    plt.title('Gender Distribution by Cluster')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plots_base64.append({'img_base64': img_base64, 'label': ''})
    plt.clf()  # Clear the current figure

    # Create treemap plots for categorical features with many values
    categorical_features = ['Subscribed Channels', 'Subscribed Account Type', 'Source of Income', 'Province']
    for feature in categorical_features:
        for cluster in data_processed['Cluster'].unique():
            cluster_data = data_processed[data_processed['Cluster'] == cluster]
            feature_counts = cluster_data[feature].value_counts()
            plt.figure(figsize=(12, 8))

            colors = plt.cm.Purples(np.linspace(0, 1, len(feature_counts)))
            squarify.plot(sizes=feature_counts.values, label=feature_counts.index, alpha=.8)
            plt.title(f'{cluster_labels[cluster]} - {feature} Distribution')
            plt.axis('off')
            
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)

            img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
            plots_base64.append({'img_base64': img_base64, 'label': cluster_labels[cluster]})
            plt.clf()  # Clear the current figure

    return jsonify({'plots': plots_base64})


if __name__ == '__main__':
    app.run(debug=True)

