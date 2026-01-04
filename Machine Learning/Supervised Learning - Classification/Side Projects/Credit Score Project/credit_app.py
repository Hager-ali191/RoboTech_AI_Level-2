# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

# Core libs
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# Sklearn bits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
)
from sklearn.feature_selection import f_classif, SelectKBest, chi2

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import joblib

# --------------------------------------------------------------------------------------
# Ocean theme CSS
# --------------------------------------------------------------------------------------
def ocean_style():
    st.markdown(
        """
        <style>
        /* Ocean gradient background */
        .stApp {
            background: linear-gradient(to bottom, #E6F7FF, #B3E5FC);
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #1CA9C9, #006994);
            color: white;
        }
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #E6F7FF;
        }
        /* Titles */
        h1, h2, h3 {
            color: #003366;
        }
        /* Buttons */
        .stButton>button {
            background-color: #1CA9C9;
            color: white;
            border-radius: 10px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #006994;
            color: #E6F7FF;
        }
        /* Selectbox and inputs */
        .stSelectbox [role="combobox"], .stNumberInput input, .stTextInput input {
            border-radius: 8px;
            border: 1px solid #1CA9C9;
        }
        /* Metrics cards */
        div[data-testid="stMetricValue"] {
            color: #006994;
            font-weight: bold;
        }
        /* Dataframes */
        .stDataFrame div {
            color: #003366;
        }
        /* Expander header */
        .st-expanderHeader {
            background: #B3E5FC;
            color: #003366;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------------------------------------------
# Your MachineLearning class (matching your notebook semantics, adapted for Streamlit)
# --------------------------------------------------------------------------------------
class MachineLearning:

    def data_information(self, data_frame):
        name_of_each_column = [col for col in data_frame]
        data_types_of_each_column = [data_frame[col].dtype for col in data_frame.columns]
        null_values_of_each_column = [data_frame[col].isnull().sum() for col in data_frame.columns]
        percentage_of_null_values_of_each_column = [data_frame[col].isnull().sum() / len(data_frame) * 100 for col in data_frame.columns]
        num_of_unique_values_of_each_column = [data_frame[col].nunique() for col in data_frame.columns]
        unique_values_of_each_column = [data_frame[col].unique() for col in data_frame.columns]
        duplicates = data_frame.duplicated().sum()

        information_of_data = pd.DataFrame({
            'Names': name_of_each_column,
            'Values': unique_values_of_each_column,
            'Data Type': data_types_of_each_column,
            'Unique Num': num_of_unique_values_of_each_column,
            'Null Num': null_values_of_each_column,
            'Null Percentage': percentage_of_null_values_of_each_column,
            'Duplicates': duplicates
        })
        return information_of_data

    # --- Plotting (return figures for Streamlit) ---
    def bar_plot(self, column, data_frame):
        sns.set_style("whitegrid")
        surviver_counts = data_frame[column].value_counts(normalize=True) * 100
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.barplot(x=surviver_counts.index, y=data_frame[column].value_counts(), palette="rocket", ax=ax)
        ax.set_title(f"Percentage of {column} and Non-{column}", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"{column} Status", fontsize=12, fontweight='bold')
        ax.set_ylabel("Count", fontsize=12, fontweight='bold')
        for p, percentage in zip(ax.patches, surviver_counts.values):
            ax.annotate(f'{percentage:.1f}%',
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
        return fig

    def histogram_plot(self, column, data_frame):
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        sk = skew(data_frame[column], nan_policy="omit")
        sns.histplot(data_frame[column], bins=50, kde=True, color="navy", edgecolor="black", ax=ax)
        ax.set_xlabel(column, fontsize=14, fontweight='bold')
        ax.set_ylabel("Frequency", fontsize=14, fontweight='bold')
        ax.set_title(f"Distribution of {column}", fontsize=16, fontweight='bold')
        ax.text(x=data_frame[column].max() * 0.7,
                y=ax.get_ylim()[1] * 0.7,
                s=f"Skewness: {sk:.2f}",
                fontsize=15, fontweight="bold", color="navy")
        return fig

    def strip_plot(self, column_x, column_y, data_frame):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.stripplot(x=data_frame[column_x], y=data_frame[column_y], jitter=True, alpha=0.7,
                      palette=["#1f77b4", "#ff7f0e"], ax=ax)
        ax.set_xlabel(column_x, fontsize=14, fontweight="bold")
        ax.set_ylabel(column_y, fontsize=14, fontweight="bold")
        ax.set_title(f"Strip Plot of {column_y} by {column_x}", fontsize=16, fontweight="bold")
        return fig

    def pie_chart(self, column, data_frame):
        counts = data_frame[column].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Dark2.colors)
        ax.set_title(f"Distribution of {column}")
        ax.axis('equal')
        return fig

    def box_plot(self, column_x, column_y, data_frame):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x=column_x, y=column_y, data=data_frame, palette="rocket", ax=ax)
        ax.set_xlabel(column_x)
        ax.set_ylabel(column_y)
        ax.set_title(f'Box Plot of {column_y} by {column_x}')
        return fig

    def count_plot(self, column, hue, data_frame):
        fig, ax = plt.subplots()
        sns.countplot(x=column, data=data_frame, palette='rocket', hue=hue, ax=ax)
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(p.get_x() + p.get_width() / 2, height,
                        f'{height/len(data_frame)*100:.2f}%', ha='center', va='bottom', fontsize=10, color='black')
        ax.set_title(f'Count of {column} with {hue} Status')
        return fig

    def scatter_plot(self, x, y, hue, data_frame):
        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y, data=data_frame, hue=hue, palette='rocket', ax=ax)
        ax.set_title(f'Scatter plot of {x} vs {y} by {hue}')
        return fig

    def heatmap(self, data_frame):
        fig, ax = plt.subplots(figsize=(9, 6))
        corr = data_frame.corr()
        sns.heatmap(corr, annot=True, cmap="rocket", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        return fig

    def violin_plot(self, x, y, hue, data_frame):
        fig, ax = plt.subplots()
        sns.violinplot(x=x, y=y, data=data_frame, hue=hue, palette='rocket', split=True, ax=ax)
        ax.set_title(f'Violin plot of {y} across {x} grouped by {hue}')
        return fig

    def swarm_plot(self, x, y, hue, data_frame):
        fig, ax = plt.subplots()
        sns.swarmplot(x=x, y=y, data=data_frame, hue=hue, palette='rocket', dodge=True, ax=ax)
        ax.set_title(f'Swarm plot of {y} across {x} grouped by {hue}')
        return fig

    def kde_plot(self, column, hue, data_frame):
        fig, ax = plt.subplots()
        sns.kdeplot(data=data_frame, x=column, hue=hue, fill=True, palette='rocket', ax=ax)
        ax.set_title(f'KDE plot of {column} by {hue}')
        return fig

    def line_plot(self, x, y, hue, data_frame):
        fig, ax = plt.subplots()
        sns.lineplot(x=x, y=y, data=data_frame, hue=hue, palette='rocket', ax=ax)
        ax.set_title(f'Line plot of {y} vs {x} by {hue}')
        return fig

    # --- Preprocessing & feature tools ---
    def handle_null_values(self, handling_type, columns, data_frame):
        df = data_frame.copy()
        if handling_type == 'mode':
            for col in columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        elif handling_type == 'knn imputer':
            for col in columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        elif handling_type == 'median':
            for col in columns:
                df[col] = df[col].fillna(df[col].median())
        return df

    def check_outliers(self, columns, data_frame, whis=1.5):
        n = len(columns)
        rows = int(np.ceil(n / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(20, 5 * rows))
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(columns):
            sns.boxplot(data=data_frame, y=col, ax=axes[i], palette='magma', whis=whis)
            axes[i].set_title(f'Boxplot of {col}', fontsize=12)
            axes[i].set_xlabel('')
            axes[i].set_ylabel(col)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        fig.tight_layout()
        return fig

    def handle_outliers(self, data_frame, column, upper_value=1.5, lower_value=1.5, handle='no'):
        df = data_frame.copy()
        logs = []
        for col in column:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (lower_value * IQR)
            upper_bound = Q3 + (upper_value * IQR)
            outliers_lower = (df[col] < lower_bound).sum()
            outliers_upper = (df[col] > upper_bound).sum()
            logs.append(f"{col}: lower={outliers_lower}, upper={outliers_upper}, lb={lower_bound:.3f}, ub={upper_bound:.3f}")
            if handle == 'yes':
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        return df, logs

    def scaling_data(self, scaler_type, data_frame, features_train, features_test, columns_list):
        if scaler_type == 'standard scaler':
            scaler = StandardScaler()
        elif scaler_type == 'min max scaler':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust scaler':
            scaler = RobustScaler()
        else:
            return features_train, features_test
        Xt = features_train.copy()
        Xv = features_test.copy()
        Xt[columns_list] = scaler.fit_transform(Xt[columns_list])
        Xv[columns_list] = scaler.transform(Xv[columns_list])
        return Xt, Xv

    def ordinal_encoding_data(self, encoding_type, features_train, features_test, data_frame, columns_list):
        if encoding_type != 'ordinal' or len(columns_list) == 0:
            return features_train, features_test
        Xt = features_train.copy()
        Xv = features_test.copy()
        all_categories = {}
        for col in columns_list:
            train_cats = Xt[col].dropna().unique()
            all_categories[col] = sorted(set(train_cats))
        ordinal_encoder = OrdinalEncoder(
            categories=[all_categories[col] for col in columns_list],
            handle_unknown='use_encoded_value',
            unknown_value=9999
        )
        Xt[columns_list] = ordinal_encoder.fit_transform(Xt[columns_list])
        Xv[columns_list] = ordinal_encoder.transform(Xv[columns_list])
        return Xt, Xv

    def spliting_data(self, data_frame, label, test_size=0.2, random_state=42):
        features = data_frame.drop([label], axis=1)
        target = data_frame[label]
        return train_test_split(features, target, test_size=test_size, random_state=random_state)

    def correlation(self, features_train, target_train, data_frame, numerical_columns, categorical_columns):
        x_num = features_train[numerical_columns]
        y = target_train
        f_values, p_values = f_classif(x_num, y)
        numerical_anova_data_frame = pd.DataFrame({
            'Feature': numerical_columns,
            'F-Score': f_values,
            'P-Value': p_values
        }).sort_values(by='F-Score', ascending=False)

        x_cat = features_train[categorical_columns].copy().fillna(0)
        for c in x_cat.columns:
            m = x_cat[c].min()
            if m < 0:
                x_cat[c] = x_cat[c] - m
        chi2_selector = SelectKBest(score_func=chi2, k='all')
        chi2_selector.fit(x_cat, y)
        categorical_chi2_data_frame = pd.DataFrame({
            'Feature': x_cat.columns,
            'Chi2 Score': chi2_selector.scores_,
            'P-Value': chi2_selector.pvalues_
        }).sort_values(by='Chi2 Score', ascending=False)
        return numerical_anova_data_frame, categorical_chi2_data_frame

    def train_test_evaluate(self, scaler_type, x_train, x_test, y_train, y_test, model):
        model.fit(x_train, y_train)
        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)
        return {
            'scaler': scaler_type,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test)
        }

# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------
st.set_page_config(page_title=" Credit Scoring Project", layout="wide")
ocean_style()
st.title("Credit Scoring Project")

# Session state
def init_state():
    defaults = {
        "df": None, "target_col": None,
        "numerical_columns": None, "categorical_columns": None,
        "amount_binner_created": False,
        "feature_engineered": False,
        "split": None,  # (X_train, X_test, y_train, y_test)
        "scaled": {},   # {"ss":(X_train_ss,X_test_ss), "mms":..., "rs":...}
        "encoded": {},  # same keys as scaled; values are (X_train_enc, X_test_enc)
        "models": {},   # metrics per model/scaler
        "fsel": None,   # (anova_df, chi2_df)
        "trained_model": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
ml = MachineLearning()

# Sidebar navigation
st.sidebar.title("🐬 Navigation")
menu = st.sidebar.radio(
    "Choose a step",
    ["Load data", "Information", "Analysis", "Preprocessing", "Feature engineering", "Splitting", "Scaling", "Encoding",
     "Feature selection", "Modeling", "Evaluation", "Export"]
)

# ---------------------------
# Load data
# ---------------------------
if menu == "Load data":
    st.subheader("🐠 Load dataset")
    uploaded = st.file_uploader("Drop your dataset here 🌊 (CSV or XLSX)", type=["csv", "xlsx"])
    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.session_state.df = df.copy()
        st.success(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head())

    if st.session_state.df is not None:
        st.markdown("### Set target column")
        try_default = list(st.session_state.df.columns).index("Class") if "Class" in st.session_state.df.columns else 0
        st.session_state.target_col = st.selectbox("Target", st.session_state.df.columns, index=try_default)

        st.markdown("### Choose initial column types (refine anytime)")
        num_guess = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        cat_guess = [c for c in st.session_state.df.columns if c not in num_guess]
        st.session_state.numerical_columns = st.multiselect("Numerical columns", st.session_state.df.columns, default=num_guess)
        st.session_state.categorical_columns = st.multiselect("Categorical columns", st.session_state.df.columns, default=[c for c in cat_guess if c != st.session_state.target_col])

# ---------------------------
# Information
# ---------------------------
elif menu == "Information":
    st.subheader("📘 Data information and stats")
    if st.session_state.df is None:
        st.warning("Load a dataset first.")
    else:
        info_df = ml.data_information(st.session_state.df)
        st.markdown("### Information table")
        st.dataframe(info_df)
        st.markdown("### describe()")
        st.dataframe(st.session_state.df.describe(include="all").transpose())
        st.markdown("### Nulls and duplicates")
        st.write(f"Total null values: {int(st.session_state.df.isnull().sum().sum())}")
        st.write(f"Total duplicates: {int(st.session_state.df.duplicated().sum())}")

# ---------------------------
# Analysis
# ---------------------------
elif menu == "Analysis":
    st.subheader("🔎 Exploratory analysis")
    if st.session_state.df is None:
        st.warning("Load a dataset first.")
    else:
        df = st.session_state.df
        plot = st.selectbox(
            "Choose a plot",
            ["Pie (Class)", "Strip (Class vs Amount)", "Box (Class vs Amount)", "Histogram (Amount)",
             "Strip (Class vs Age)", "Box (Class vs Age)", "Histogram (Age)",
             "Box (Class vs Duration)", "Histogram (Duration)",
             "Scatter (Amount vs Age by Class)", "Scatter (Amount vs Duration by Class)",
             "Count/KDE/Pie — any column"]
        )

        if plot == "Pie (Class)":
            fig = ml.pie_chart('Class', df); st.pyplot(fig)
        elif plot == "Strip (Class vs Amount)":
            fig = ml.strip_plot('Class', 'Amount', df); st.pyplot(fig)
        elif plot == "Box (Class vs Amount)":
            fig = ml.box_plot('Class', 'Amount', df); st.pyplot(fig)
        elif plot == "Histogram (Amount)":
            fig = ml.histogram_plot('Amount', df); st.pyplot(fig)
        elif plot == "Strip (Class vs Age)":
            fig = ml.strip_plot('Class', 'Age', df); st.pyplot(fig)
        elif plot == "Box (Class vs Age)":
            fig = ml.box_plot('Class', 'Age', df); st.pyplot(fig)
        elif plot == "Histogram (Age)":
            fig = ml.histogram_plot('Age', df); st.pyplot(fig)
        elif plot == "Box (Class vs Duration)":
            fig = ml.box_plot('Class', 'Duration', df); st.pyplot(fig)
        elif plot == "Histogram (Duration)":
            fig = ml.histogram_plot('Duration', df); st.pyplot(fig)
        elif plot == "Scatter (Amount vs Age by Class)":
            fig = ml.scatter_plot('Amount', 'Age', 'Class', df); st.pyplot(fig)
        elif plot == "Scatter (Amount vs Duration by Class)":
            fig = ml.scatter_plot('Amount', 'Duration', 'Class', df); st.pyplot(fig)
        elif plot == "Count/KDE/Pie — any column":
            col = st.selectbox("Column", [c for c in df.columns if c != st.session_state.target_col])
            with st.expander("Count plot"):
                fig = ml.count_plot(col, st.session_state.target_col, df); st.pyplot(fig)
            with st.expander("KDE"):
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig = ml.kde_plot(col, st.session_state.target_col, df); st.pyplot(fig)
                else:
                    st.info("KDE requires numeric columns.")
            with st.expander("Pie"):
                fig = ml.pie_chart(col, df); st.pyplot(fig)

# ---------------------------
# Preprocessing
# ---------------------------
elif menu == "Preprocessing":
    st.subheader("🧼 Nulls, outliers, and Amount Binner")
    if st.session_state.df is None:
        st.warning("Load a dataset first.")
    else:
        df = st.session_state.df

        st.markdown("### Handle nulls")
        null_cols = st.multiselect("Columns to handle nulls", df.columns)
        strategy = st.selectbox("Strategy", ["mode", "median", "knn imputer"])
        if st.button("Apply null handling"):
            df = ml.handle_null_values(strategy, null_cols, df.copy())
            st.session_state.df = df
            st.success(f"Applied {strategy} to {len(null_cols)} columns.")
            st.dataframe(ml.data_information(df))

        st.markdown("---")
        st.markdown("### Outliers — visualize and (optionally) cap via IQR")
        out_cols = st.multiselect("Numeric columns to inspect", df.select_dtypes(include=np.number).columns.tolist(),
                                  default=[c for c in ['Duration', 'Amount', 'Age'] if c in df.columns])
        whis = st.slider("Whisker multiplier (IQR)", 1.0, 5.0, 1.5, 0.1)
        cap = st.checkbox("Cap outliers to bounds", value=False)
        if st.button("Check/Handle outliers"):
            if out_cols:
                fig = ml.check_outliers(out_cols, df, whis=whis); st.pyplot(fig)
                df2, logs = ml.handle_outliers(df.copy(), out_cols, upper_value=whis, lower_value=whis, handle='yes' if cap else 'no')
                st.session_state.df = df2
                st.markdown("#### Outlier counts and bounds per column")
                for line in logs: st.write(line)
                st.success("Outlier step complete.")

        st.markdown("---")
        st.markdown("### Amount Binner — your custom binning and distribution")
        def amount_binner(x, bin_size=1000):
            return int((x - 1) // bin_size + 1)

        bin_size = st.number_input("Bin size", min_value=100, value=1000, step=100)
        if st.button("Create Amount Binner"):
            if 'Amount' not in df.columns:
                st.error("Amount column not found.")
            else:
                df['Amount Binner'] = df['Amount'].apply(lambda x: amount_binner(x, bin_size))
                st.session_state.df = df
                st.session_state.amount_binner_created = True
                st.success("Amount Binner created.")
                st.write("Unique bins:", sorted(df['Amount Binner'].unique()))
                duplicated = {}
                for value in df['Amount Binner']:
                    duplicated[value] = duplicated.get(value, 0) + 1
                percentage = {k: (v / len(df)) * 100 for k, v in duplicated.items()}
                st.markdown("#### Bin counts and percentages")
                st.dataframe(pd.DataFrame({"bin": list(duplicated.keys()), "count": list(duplicated.values()),
                                           "percent": [percentage[k] for k in duplicated.keys()]}).sort_values("bin"))
                fig = ml.pie_chart('Amount Binner', df); st.pyplot(fig)

        st.markdown("---")
        st.markdown("### Extra outlier checks for Amount Binner and Age")
        if st.session_state.amount_binner_created:
            with st.expander("Amount Binner (whis=5)"):
                fig = ml.check_outliers(['Amount Binner'], st.session_state.df, whis=5); st.pyplot(fig)
        if 'Age' in df.columns:
            with st.expander("Age (whis=2.2)"):
                fig = ml.check_outliers(['Age'], st.session_state.df, whis=2.2); st.pyplot(fig)

# ---------------------------
# Feature engineering
# ---------------------------
elif menu == "Feature engineering":
    st.subheader("🧪 Create ratios and composites")
    if st.session_state.df is None:
        st.warning("Load a dataset first.")
    else:
        df = st.session_state.df.copy()

        add_amount_duration = st.checkbox("Amount Duration Ratio = Amount / Duration", True)
        add_age_per_duration = st.checkbox("Age Per Duration = Age / Duration", True)
        add_checking_history = st.checkbox("Checking History = Checking + History", True)
        add_savings_property = st.checkbox("Savings Property = Savings + Property", True)
        add_amount_age = st.checkbox("Amount Age Ratio = Amount / Age", True)
        add_duration_age = st.checkbox("Duration Age Ratio = Duration / Age", True)

        if st.button("Apply feature engineering"):
            created = []
            if add_amount_duration and {'Amount', 'Duration'}.issubset(df.columns):
                df['Amount Duration Ratio'] = df['Amount'] / df['Duration']; created.append('Amount Duration Ratio')
            if add_age_per_duration and {'Age', 'Duration'}.issubset(df.columns):
                df['Age Per Duration'] = df['Age'] / df['Duration']; created.append('Age Per Duration')
            if add_checking_history and {'Checking', 'History'}.issubset(df.columns):
                df['Checking History'] = df['Checking'] + df['History']; created.append('Checking History')
            if add_savings_property and {'Savings', 'Property'}.issubset(df.columns):
                df['Savings Property'] = df['Savings'] + df['Property']; created.append('Savings Property')
            if add_amount_age and {'Amount', 'Age'}.issubset(df.columns):
                df['Amount Age Ratio'] = df['Amount'] / df['Age']; created.append('Amount Age Ratio')
            if add_duration_age and {'Duration', 'Age'}.issubset(df.columns):
                df['Duration Age Ratio'] = df['Duration'] / df['Age']; created.append('Duration Age Ratio')

            st.session_state.df = df
            st.session_state.feature_engineered = True
            st.success(f"Created: {', '.join(created) if created else 'No features created (missing columns).'}")
            if created:
                st.dataframe(df[[c for c in df.columns if c in created]].head())

# ---------------------------
# Splitting
# ---------------------------
elif menu == "Splitting":
    st.subheader("✂️ Train/test split")
    if st.session_state.df is None or st.session_state.target_col is None:
        st.warning("Load data and set target.")
    else:
        df = st.session_state.df
        target = st.session_state.target_col
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", 0, 9999, 42, 1)
        if st.button("Split"):
            X_train, X_test, y_train, y_test = ml.spliting_data(df, target, test_size=float(test_size), random_state=int(random_state))
            st.session_state.split = (X_train, X_test, y_train, y_test)
            st.success("Data split.")
            st.write("X_train:", X_train.shape, "X_test:", X_test.shape, "y_train:", y_train.shape, "y_test:", y_test.shape)

# ---------------------------
# Scaling
# ---------------------------
elif menu == "Scaling":
    st.subheader("📏 Standard / MinMax / Robust")
    if st.session_state.split is None:
        st.warning("Split data first.")
    else:
        X_train, X_test, y_train, y_test = st.session_state.split
        num_cols = st.multiselect("Numeric columns to scale", X_train.select_dtypes(include=np.number).columns.tolist(),
                                  default=[c for c in ['Duration', 'Age', 'Amount', 'Amount Duration Ratio', 'Amount Age Ratio'] if c in X_train.columns])
        if st.button("Apply scaling"):
            X_train_ss, X_test_ss = ml.scaling_data('standard scaler', st.session_state.df, X_train.copy(), X_test.copy(), num_cols)
            X_train_mms, X_test_mms = ml.scaling_data('min max scaler', st.session_state.df, X_train.copy(), X_test.copy(), num_cols)
            X_train_rs, X_test_rs = ml.scaling_data('robust scaler', st.session_state.df, X_train.copy(), X_test.copy(), num_cols)
            st.session_state.scaled = {"ss": (X_train_ss, X_test_ss), "mms": (X_train_mms, X_test_mms), "rs": (X_train_rs, X_test_rs)}
            st.success("Scaling applied (SS, MMS, RS).")
            with st.expander("Preview scaled columns (SS)"):
                st.dataframe(X_train_ss[num_cols].head())
            with st.expander("Preview scaled columns (MMS)"):
                st.dataframe(X_train_mms[num_cols].head())
            with st.expander("Preview scaled columns (RS)"):
                st.dataframe(X_train_rs[num_cols].head())

# ---------------------------
# Encoding
# ---------------------------
elif menu == "Encoding":
    st.subheader("🔤 Ordinal + one-hot")
    if not st.session_state.scaled:
        st.warning("Apply scaling first.")
    else:
        label_or_onehot = st.multiselect("Ordinal columns", st.session_state.df.columns.tolist(), default=['Purpose'] if 'Purpose' in st.session_state.df.columns else [])
        onehot_future = st.multiselect("One-hot columns", st.session_state.df.columns.tolist(),
                                       default=[c for c in ['Foreign', 'Telephone', 'Depends', 'housing', 'Other', 'Coapp'] if c in st.session_state.df.columns])

        encoded = {}
        X_train, X_test, y_train, y_test = st.session_state.split

        if st.button("Apply encoding on all scaled sets"):
            for key, (Xt, Xv) in st.session_state.scaled.items():
                Xt_e, Xv_e = ml.ordinal_encoding_data("ordinal", Xt.copy(), Xv.copy(), st.session_state.df, label_or_onehot)
                if onehot_future:
                    Xt_e = pd.get_dummies(Xt_e, columns=[c for c in onehot_future if c in Xt_e.columns])
                    Xv_e = pd.get_dummies(Xv_e, columns=[c for c in onehot_future if c in Xv_e.columns])
                Xt_e, Xv_e = Xt_e.align(Xv_e, join="outer", axis=1, fill_value=0)
                bool_cols = Xt_e.select_dtypes(include="bool").columns
                Xt_e[bool_cols] = Xt_e[bool_cols].astype(int)
                Xv_e[bool_cols] = Xv_e[bool_cols].astype(int)
                encoded[key] = (Xt_e, Xv_e)
            st.session_state.encoded = encoded
            st.success("Encoding applied and aligned for SS/MMS/RS.")
            with st.expander("Encoded preview (SS)"):
                st.dataframe(encoded.get("ss", (pd.DataFrame(),))[0].head())

# ---------------------------
# Feature selection
# ---------------------------
elif menu == "Feature selection":
    st.subheader("🧠 ANOVA & Chi² ranking")
    if not st.session_state.encoded:
        st.warning("Complete scaling + encoding first.")
    else:
        variant = st.selectbox("Variant", ["ss", "mms", "rs"], index=0)
        X_train, X_test, y_train, y_test = st.session_state.split
        X_train_e, X_test_e = st.session_state.encoded[variant]

        all_num = X_train_e.select_dtypes(include=np.number).columns.tolist()
        default_nums = [c for c in ['Duration', 'Amount', 'Age', 'Amount Duration Ratio', 'Age Per Duration', 'Amount Age Ratio', 'Duration Age Ratio', 'Amount Binner'] if c in all_num]
        num_cols = st.multiselect("Numerical columns", all_num, default=default_nums)
        cat_cols = st.multiselect("Categorical columns", [c for c in X_train_e.columns if c not in num_cols],
                                  default=[c for c in X_train_e.columns if c not in num_cols][:10])

        if st.button("Run feature selection"):
            anova_df, chi2_df = ml.correlation(X_train_e, y_train, st.session_state.df, num_cols, cat_cols)
            st.session_state.fsel = (anova_df, chi2_df)
            st.markdown("### ANOVA (numerical)")
            st.dataframe(anova_df)
            st.markdown("### Chi-square (categorical)")
            st.dataframe(chi2_df)

# ---------------------------
# Modeling
# ---------------------------
elif menu == "Modeling":
    st.subheader("🚀 Train LR/KNN/SVM/DT/RF/XGB/GB across scalers")
    if not st.session_state.encoded:
        st.warning("Complete scaling + encoding first.")
    else:
        X_train, X_test, y_train, y_test = st.session_state.split
        models_to_run = st.multiselect("Models", ["Logistic Regression", "KNN (Manhattan)", "KNN (Euclidean)", "SVM (RBF)", "SVM (Linear)", "Decision Tree", "Random Forest", "XGBoost", "Gradient Boosting"],
                                       default=["Logistic Regression", "KNN (Manhattan)", "SVM (RBF)", "Decision Tree", "Random Forest", "XGBoost", "Gradient Boosting"])
        lr = LogisticRegression(random_state=42, C=10, penalty='l2', solver="newton-cg", max_iter=1000)
        knn_manhattan = KNeighborsClassifier(n_neighbors=17, metric='manhattan')
        knn_euclidean = KNeighborsClassifier(n_neighbors=17, metric='euclidean')
        svm_rbf = SVC(C=1, kernel='rbf', random_state=42, probability=True)
        svm_lin = SVC(C=1, kernel='linear', random_state=42, probability=True)
        dt = DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=7, min_samples_leaf=5, min_samples_split=10)
        rf = RandomForestClassifier(random_state=42, n_estimators=100, criterion='gini', max_depth=50, min_samples_leaf=4, min_samples_split=4)
        xgbc = xgb.XGBClassifier(random_state=42, learning_rate=0.01, max_depth=3, n_estimators=250, eval_metric="logloss")
        gbc = GradientBoostingClassifier(random_state=42, learning_rate=0.01, max_depth=3, n_estimators=300, min_samples_leaf=4, min_samples_split=2)

        model_map = {
            "Logistic Regression": lr,
            "KNN (Manhattan)": knn_manhattan,
            "KNN (Euclidean)": knn_euclidean,
            "SVM (RBF)": svm_rbf,
            "SVM (Linear)": svm_lin,
            "Decision Tree": dt,
            "Random Forest": rf,
            "XGBoost": xgbc,
            "Gradient Boosting": gbc
        }

        if st.button("Train across SS/MMS/RS"):
            results = {}
            for scaler_key in ["ss", "mms", "rs"]:
                if scaler_key not in st.session_state.encoded: continue
                Xt, Xv = st.session_state.encoded[scaler_key]
                results[scaler_key] = {}
                for name in models_to_run:
                    model = model_map[name]
                    metrics = ml.train_test_evaluate(scaler_key.upper(), Xt, Xv, y_train, y_test, model)
                    results[scaler_key][name] = metrics
            st.session_state.models = results
            st.success("Training complete. Metrics below.")
            for scaler_key, models in results.items():
                st.markdown(f"### {scaler_key.upper()} results")
                st.dataframe(pd.DataFrame(models).transpose())

            # Save a trained model for evaluation tab (RF on SS if available)
            if "ss" in st.session_state.encoded and "Random Forest" in models_to_run:
                model = rf.fit(st.session_state.encoded["ss"][0], y_train)
                st.session_state.trained_model = ("Random Forest", model, "ss")

# ---------------------------
# Evaluation
# ---------------------------
elif menu == "Evaluation":
    st.subheader("📈 Confusion, report, ROC, PR")
    if st.session_state.trained_model is None:
        st.warning("Train at least one model in Modeling tab.")
    else:
        name, model, scaler_key = st.session_state.trained_model
        X_train, X_test, y_train, y_test = st.session_state.split
        Xt, Xv = st.session_state.encoded[scaler_key]

        y_pred = model.predict(Xv)
        acc = accuracy_score(y_test, y_pred)
        prec_w = precision_score(y_test, y_pred, average="weighted")
        rec_w = recall_score(y_test, y_pred, average="weighted")
        f1_w = f1_score(y_test, y_pred, average="weighted")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision (weighted)", f"{prec_w:.3f}")
        c3.metric("Recall (weighted)", f"{rec_w:.3f}")
        c4.metric("F1 (weighted)", f"{f1_w:.3f}")

        st.markdown("### Classification report")
        st.code(classification_report(y_test, y_pred))

        st.markdown("### Confusion matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

