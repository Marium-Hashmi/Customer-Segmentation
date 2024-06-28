# Customer-Segmentation

## Project Overview
This project aims to provide a comprehensive analysis of customer segmentation and risk management in the banking sector. Using clustering techniques such as K-means, this project segments customers into distinct risk groups based on various features. It also provides insightful visualizations to help understand the risk profiles of different customer segments.

## Table of Contents
- Project Overview
- Features
- Installation
- Usage
- Data Preprocessing
- Clustering
- Visualization
- Results

## Features
- **Data Ingestion and Preprocessing:**

Robust data ingestion methods to load customer data from various sources.
Preprocessing of data to handle missing values using K-nearest neighbors (KNN) imputation and encoding categorical data for analysis.
- **Dimensionality Reduction:**

Integration of Principal Component Analysis (PCA) to reduce the dimensionality of the data, enhancing the performance of clustering algorithms.
- **Customer Segmentation:**

Utilization of K-means clustering to segment customers into distinct risk groups.
Evaluation of clustering performance using intrinsic measures like silhouette score and Davies-Bouldin index.
- **Visualization and Reporting:**

Development of interactive visualizations to represent clustering results and customer segmentation.
Generation of comprehensive reports highlighting the characteristics and behaviors of each customer segment.
- **API Integration:**

Creation of APIs for seamless integration with existing banking systems and workflows, enabling real-time risk analysis and decision-making.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Marium-Hashmi/Customer-Segmentation.git
    cd Customer-Segmentation
    ```

2. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Data Preprocessing and Clustering**:
    - Run the `main.py` script to perform data preprocessing and clustering:
      ```sh
      python main.py
      ```
    - Upload RiskSummary excel.
    - Perform preprocessing, clustering & see results.

3. **API Integration**:
    - Push Customer Data to check risk cluster for given customer.

4. **Visualization**:
    - Use the Results & Visualization tab to see results cluster wise.

## Data Preprocessing

- **Ingestion**: Load customer data from various sources.
- **Preprocessing**: Clean data by removing duplicates, correcting errors, and handling missing values using KNN imputation.

## Clustering

- **K-means Clustering**: Segment customers into distinct risk groups.
- **Evaluation Metrics**: Evaluate clustering performance using silhouette score and Davies-Bouldin index.

## Visualization

- **Interactive Visualizations**: Represent clustering results and customer segmentation using various plots and graphs.
- **Treemaps**: Display distribution of categorical features within clusters.
- **Bar Graphs and Line Charts**: Show numerical feature distributions across clusters.

## Results

### Clustering Insights and Implications for Risk Management in Banking

The clustering analysis provided several valuable insights into the customer base and their associated risk levels. By segmenting customers into distinct clusters, we were able to identify patterns and characteristics that are crucial for effective risk management.

#### Key Insights

1. **Identification of High-Risk Customers**: Customers with active PEP status and higher transaction volumes.
2. **Segmentation of Medium-Risk Customers**: Customers with moderate transaction volumes.
3. **Identification of Low-Risk Customers**: Customers with inactive PEP status and lower transaction volumes.
4. **Customized Financial Products**: Design tailored financial products based on customer risk profiles.
5. **Enhanced Risk Monitoring**: Continuous monitoring and dynamic updates to clustering models.
6. **Improved Customer Retention**: Provide personalized services to enhance customer satisfaction and retention.
