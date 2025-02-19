import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Set page config
st.set_page_config(page_title="Clustering App", layout="wide")

# Define the CSS for the background image
background_image_url = "https://miro.medium.com/v2/resize:fit:1000/1*wvb3iMZedNO54vkyoCw3-w.jpeg"  # Replace with your image URL

page_bg_img = f"""
<style>
    body {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    
    /* Add background to sidebar */
    [data-testid="stSidebar"] {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background: rgba(255, 255, 255, 0.8); /* Slight white transparency */
        background-position: center;
        background-repeat: no-repeat;
    }}

    /* Optional: Make text more readable */
    .stApp {{
        background-color: rgba(255, 255, 255, 0.7); /* White background with transparency */
        padding: 20px;
        border-radius: 10px;
    }}
</style>
"""

# Inject CSS into the Streamlit app
st.markdown(page_bg_img, unsafe_allow_html=True)


# Streamlit UI
st.title("Clustering Analysis Web Application")


# File uploader
data_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if data_file:
    if "World_development_mesurement" not in data_file.name:
        st.error("Please upload the correct 'World Development Measurement' dataset.")
    else:
        st.success("Correct file uploaded. Proceed with the analysis.")
    
    df = None # Prevents further processing

    if data_file:   # Only execute if a file is uploaded
        try:
            xl = pd.ExcelFile(data_file)
            if "world_development" not in xl.sheet_names:
                st.error("The uploaded file does not contain the required 'world_development' sheet. Please upload the correct file.")
            else:
                df = xl.parse("world_development")

        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
    
    # Only proceed if df is successfully loaded
    if df is not None: 
        # Display the dataset
        if st.checkbox("### Dataset Preview"):
            st.write(df)

            # Display Shape
            st.write(f"Shape of the dataset: {df.shape}")

        # Encode categorical column 'Country'
        if 'Country' in df.columns:
           le = LabelEncoder()
           df['Country'] = le.fit_transform(df['Country'])
    
        # Convert monetary columns to numeric
        columns_to_clean = ["GDP", "Health Exp/Capita", "Tourism Inbound", "Tourism Outbound"]
        for col in columns_to_clean:
            if col in df.columns:
                df[col] = df[col].replace({"[$,]": ""}, regex=True).astype(float)
    
        # Convert percentage columns to numeric
        df = df.replace({'%': ''}, regex=True)
        df = df.apply(pd.to_numeric, errors='coerce')

        # Drop columns with more than 40% missing values
        threshold = 0.4 * len(df)
        df = df.dropna(axis=1, thresh=threshold)

        # Impute remaining missing values with median
        imputer = SimpleImputer(strategy="median")
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        # Rename cleaned data
        df_cleaned = df.copy()
    
        # Scale the data
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df.columns)

        # Checkbox to show cleaned data
        if st.checkbox("Show Cleaned Data"):
            st.write(df_cleaned)

        # Visualization Checkbox
        if st.checkbox("Show Data Visualizations"):
            st.markdown('<h4 style="color: navy;">Feature Distributions: </h4>', unsafe_allow_html=True)
            fig, axes = plt.subplots(figsize=(12, 12))
            df_scaled.hist(ax=axes, bins=30)
            plt.suptitle("Feature Distributions", fontsize=14)
            st.pyplot(fig)

        # Sidebar: Feature Relationship Scatter Plot
        st.sidebar.markdown('<h1 style="color: gray;">Feature Relationship Analysis: </h1>', unsafe_allow_html=True)
        feature_x = st.sidebar.selectbox("Select X-axis Feature", df_scaled.columns)
        feature_y = st.sidebar.selectbox("Select Y-axis Feature", df_scaled.columns)

        if st.sidebar.button("Show Scatter Plot"):
            fig, ax = plt.subplots()
            sns.scatterplot(x=df_scaled[feature_x], y=df_scaled[feature_y], alpha=0.6)
            plt.xlabel(feature_x)
            plt.ylabel(feature_y)
            plt.title(f"Relationship between {feature_x} and {feature_y}")
            st.sidebar.pyplot(fig)

            # Text Analysis of Relationship
            correlation = df_scaled[feature_x].corr(df_scaled[feature_y])
            if correlation > 0.7:
                st.sidebar.write(f"**Strong positive correlation** between {feature_x} and {feature_y}.")
            elif correlation > 0.3:
                st.sidebar.write(f"**Moderate positive correlation** between {feature_x} and {feature_y}.")
            elif correlation > -0.3:
                st.sidebar.write(f"**Weak or no correlation** between {feature_x} and {feature_y}.")
            elif correlation > -0.7:
                st.sidebar.write(f"**Moderate negative correlation** between {feature_x} and {feature_y}.")
            else:
                st.sidebar.write(f"**Strong negative correlation** between {feature_x} and {feature_y}.")
        
        # Feature Selection with all available columns
        all_columns = df_scaled.columns.tolist()
        selected_features = st.multiselect("Select at least two Columns for Clustering", all_columns, default=all_columns[:0])
    
        if len(selected_features) < 2:
            st.warning("Please select at least two Columns.")
        else:
           df_selected = df_scaled[selected_features]

    if selected_features:
        # User selects number of clusters
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=4)
    
        # Model selection
        model_option = st.selectbox("Select Clustering Model", ["K-Means", "Agglomerative Clustering", "DBSCAN", "Gaussian Mixture Model"])
        
        if 'model_performance' not in st.session_state:
            st.session_state['model_performance'] = {}
        

        # Train the model
        if st.button("Train And Predict"):
            labels = None
        
            if model_option == "K-Means":
                model = KMeans(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(df_selected)
            elif model_option == "Agglomerative Clustering":
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(df_selected)
            elif model_option == "DBSCAN":
                model = DBSCAN(eps=0.1, min_samples=5)
                labels = model.fit_predict(df_selected)
                if len(set(labels)) == 1:  # Only noise detected
                    st.error("DBSCAN detected all points as noise. Try adjusting parameters.")
                    labels = np.array(["Noise"] * len(df_selected))
            elif model_option == "Gaussian Mixture Model":
                model = GaussianMixture(n_components=n_clusters, random_state=42)
                labels = model.fit_predict(df_selected)

            st.success(f"{model_option} model trained with {n_clusters} clusters!")
        
            # Evaluation Metrics
            if len(set(labels)) > 1:
                silhouette = silhouette_score(df_selected, labels)
                db_score = davies_bouldin_score(df_selected, labels)
                ch_score = calinski_harabasz_score(df_selected, labels)
            else:
                silhouette, db_score, ch_score = np.nan, np.nan, np.nan
            
            st.session_state['model_performance'][model_option] = {
                "Silhouette Score": silhouette,
                "Davies-Bouldin Score": db_score,
                "Calinski-Harabasz Score": ch_score

            }

            st.write(f"**Silhouette Score:** {silhouette:.4f}")
            st.write(f"**Davies-Bouldin Score:** {db_score:.4f}")
            st.write(f"**Calinski-Harabasz Score:** {ch_score:.4f}")
        
            # PCA for visualization
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(df_selected)
            df_pca = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])
            df_pca['Cluster'] = labels
        
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Cluster', palette='deep', alpha=0.6)
            plt.title(f"{model_option} Clustering Visualization for features: {', '.join(df_selected.columns)}")
            st.pyplot(fig)

            # Convert labels back to categorical country names
            df_results = pd.DataFrame({"Country": le.inverse_transform(df["Country"].astype(int)), "Cluster": labels})
            st.write("### Clustered Countries", df_results)

        # Model Comparison
        if st.checkbox("Compare Models"):
            if st.session_state['model_performance']:
                comparison_metrics = {
                    model: {
                        key: (value if not isinstance(value, (np.ndarray, tuple)) else "N/A")
                        for key, value in metrics.items()
                    }
                    for model, metrics in st.session_state['model_performance'].items()
                }
                
                performance_df = pd.DataFrame(comparison_metrics).T
                performance_df.reset_index(inplace=True)
                performance_df.rename(columns={'index': 'Model'}, inplace=True)
                
                st.subheader("Model Comparison Table")
                st.write(performance_df)
                
                st.subheader("Comparison Bar Chart")
                metric_to_compare = st.selectbox("Select Metric for Comparison", ["Silhouette Score", "Davies-Bouldin Score", "Calinski-Harabasz Score"])
                
                fig, ax = plt.subplots()
                sns.barplot(x='Model', y=metric_to_compare, data=performance_df, ax=ax, palette="viridis")
                ax.set_ylabel(metric_to_compare)
                ax.set_title(f"Model Comparison: {metric_to_compare}")
                st.pyplot(fig)
                
                best_model_row = performance_df.loc[performance_df[metric_to_compare].idxmax()]
                best_model_name = best_model_row['Model']
                best_model_value = best_model_row[metric_to_compare]
                
                st.success(f"The best-performing model based on **{metric_to_compare}** is **{best_model_name}** with a score of **{best_model_value:.2f}**.")

        # Dendrogram for Hierarchical Clustering
        if model_option == "Agglomerative Clustering" and st.checkbox("Show Dendrogram"):
            truncate_level = st.slider("Select Dendrogram Truncate Level", 1, 20, 5)
            fig, ax = plt.subplots(figsize=(10, 5))
            linkage_matrix = linkage(df_scaled, method='ward')
            dendrogram(linkage_matrix, truncate_mode="level", p=truncate_level) # User-selected truncation level
            plt.title("Dendrogram for Hierarchical Clustering")
            plt.xlabel("Data Points")
            plt.ylabel("Distance")
            st.pyplot(fig)

# Training process and visualizations go here...
else:
    st.warning("Please upload World_Development_Measurement Excel file to proceed.")