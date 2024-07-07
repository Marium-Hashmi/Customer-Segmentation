# Customer-Segmentation

## Table of Contents
- Project Summary
- Novelty of The Project
- Features
- Installation
- Usage
- Data Preprocessing
- Clustering
- Visualization
- Results

## Project Summary
This project aims to provide a comprehensive analysis of customer segmentation and risk management in the banking sector. Using clustering techniques such as K-means, this project segments customers into distinct risk groups based on various features. It also provides insightful visualizations to help understand the risk profiles of different customer segments.

## Novelty of the Project
The novelty of this project lies in its integrated approach combining multi-step data preprocessing, dimensionality reduction using PCA, and clustering with K-means. Additionally, the project features a comprehensive visualization suite and API integration for real-time data processing. This holistic approach not only ensures the accuracy and usability of the data but also provides actionable insights for risk management.

## Design Diagram
![image](https://github.com/Marium-Hashmi/Customer-Segmentation/assets/33281835/3a92c8a5-c14e-44c9-822b-38512ccf3555)


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
### High-Risk Cluster - Subscribed Account Type Distribution
![AccountHigh](https://github.com/Marium-Hashmi/Customer-Segmentation/assets/33281835/7cfe6b05-89c1-4c3c-b2f3-e8fa03a2d5c6)
Customers in the high-risk cluster predominantly hold FIRST Current Accounts and show significant activity in First Daily Profit Accounts.

### Low-Risk Cluster - Subscribed Account Type Distribution
![AccountLow](https://github.com/Marium-Hashmi/Customer-Segmentation/assets/33281835/9c570040-4b5e-4034-914b-d231ee91e5c5)
Low-risk customers are more likely to have Aasan Accounts and First Pension Accounts, indicating conservative financial behavior.

### Medium-Risk Cluster - Subscribed Account Type Distribution
![AccountMedium](https://github.com/Marium-Hashmi/Customer-Segmentation/assets/33281835/294bb00c-2abe-4858-bfb0-6e6a2067212d)
Medium-risk customers hold a balanced mix of account types, including FIRST Current Accounts and Repa Accounts.

### High-Risk Cluster - Subscribed Channels Distribution
![ChannelsHigh](https://github.com/Marium-Hashmi/Customer-Segmentation/assets/33281835/96559b46-ce6b-4ba6-8b4b-18835f54cc55)
High-risk customers predominantly use ATMs and Mobile Apps, suggesting a preference for digital and quick access.

### Low-Risk Cluster - Subscribed Channels Distribution
![ChannelsLow](https://github.com/Marium-Hashmi/Customer-Segmentation/assets/33281835/298abaa6-eca8-4ab3-9191-d3a5966eb8f3)
Low-risk customers also prefer ATMs but show less digital engagement compared to high-risk customers.

### Medium-Risk Cluster - Subscribed Channels Distribution
![ChannelsMedium](https://github.com/Marium-Hashmi/Customer-Segmentation/assets/33281835/34398032-90e5-4c1c-b8be-6f059262703b)
Medium-risk customers display a balanced usage of ATMs and Mobile Apps, indicating moderate digital engagement.

### High-Risk Cluster - Province Distribution
![ProvinceHigh](https://github.com/Marium-Hashmi/Customer-Segmentation/assets/33281835/f159c672-6090-40f7-b47b-70f5752ede94)
High-risk customers are mainly concentrated in Punjab, followed by Sindh, indicating regional risk concentration.

### Low-Risk Cluster - Province Distribution
![ProvicneLow](https://github.com/Marium-Hashmi/Customer-Segmentation/assets/33281835/29dda525-67c4-4567-b92f-8d33efb09cc9)
Low-risk customers are also primarily from Punjab but show more distribution across other provinces.

### Medium-Risk Cluster - Province Distribution
![ProvinceMedium](https://github.com/Marium-Hashmi/Customer-Segmentation/assets/33281835/949b67c1-010d-4bc8-bfc2-f35efb3d1853)
Medium-risk customers are predominantly from Punjab, with significant representation in Sindh and other regions.

### High-Risk Cluster - Source of Income Distribution
![SrcIncomeHigh](https://github.com/Marium-Hashmi/Customer-Segmentation/assets/33281835/dd935935-7bb0-4001-bc48-c1be2099bcdb)
High-risk customers have diverse income sources, including self-employment in enterprises, agriculture, and livestock, indicating varied financial stability.

### Low-Risk Cluster - Source of Income Distribution
![SrcIncomeLow](https://github.com/Marium-Hashmi/Customer-Segmentation/assets/33281835/5d91f326-e9fd-4513-8c03-619bb92200da)
Low-risk customers primarily rely on pensions, salaries, and livestock, reflecting stable and predictable income sources.

### Medium-Risk Cluster - Source of Income Distribution
![SrcIncomeMedium](https://github.com/Marium-Hashmi/Customer-Segmentation/assets/33281835/4d92bcb5-8c73-4141-8c0e-61dedcf5d874)
Medium-risk customers show a mix of self-employment in enterprises and livestock, with notable representation in agriculture and other sectors.
