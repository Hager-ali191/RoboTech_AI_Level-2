
# machine_learning_class.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.feature_selection import f_classif, chi2, SelectKBest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    confusion_matrix, classification_report
)

class MachineLearning:
    """Your Machine Learning class implementation"""
    
    def data_information(self, data_frame):
        """
        Steps of understanding data to build the model ?
        
          Analysis :
            - plots of data frame for categorical & numerical
            - Some Math -> describe()
        
          Preprocessing :
            - Names of columns
            - Data types -> info() or here
            - nulls
            - duplicates
            - unique (num & values)
            - mean & median & mode : -> describe()
            - outliers -> box plot & math ( describe() )
        """
        
        """
        Just to know the information of each column.
        I get each name of it.
        Also each data type.
        """
        name_of_each_column = [col for col in data_frame]
        data_types_of_each_column = [data_frame[col].dtype for col in data_frame.columns]
        
        """
        Number of null values in each column is not enough to know if it is huge or not.
        I need to calculate the percentage of it based on length of data frame to make it clear.
        """
        null_values_of_each_column = [data_frame[col].isnull().sum() for col in data_frame.columns]
        percentage_of_null_values_of_each_column = [data_frame[col].isnull().sum() / len(data_frame) * 100 
                                                   for col in data_frame.columns]
        
        """
        Unique values make me know :
        
          - What are the exact values of each column ?
          - Column is categorical or numerical ?
        """
        num_of_unique_values_of_each_column = [data_frame[col].nunique() for col in data_frame.columns]
        unique_values_of_each_column = [data_frame[col].unique() for col in data_frame.columns]
        
        """
        What are the num of duplicates in data frame ?
        """
        duplicates = data_frame.duplicated().sum()
        
        information_of_data = pd.DataFrame(
            {
                'Names': name_of_each_column,
                'Values': unique_values_of_each_column,
                'Data Type': data_types_of_each_column,
                'Unique Num': num_of_unique_values_of_each_column,
                'Null Num': null_values_of_each_column,
                'Null Percentage': percentage_of_null_values_of_each_column,
                'Duplicates': duplicates
            }
        )
        
        return information_of_data
    
    def bar_plot(self, column, data_frame):
        sns.set_style("whitegrid")
        surviver_counts = data_frame[column].value_counts(normalize=True)*100
        
        plt.figure(figsize=(6, 6))
        ax = sns.barplot(x=surviver_counts.index, y=data_frame[column].value_counts(), palette="rocket")
        
        plt.title(f"Percentage of {column} and Non-{column}", fontsize=14, fontweight='bold')
        plt.xlabel(f"{column} Status", fontsize=12, fontweight='bold')
        plt.ylabel("Count", fontsize=12, fontweight='bold')
        
        for p, percentage in zip(ax.patches, surviver_counts.values):
            ax.annotate(f'{percentage:.1f}%',
                       (p.get_x() + p.get_width() / 2, p.get_height()),
                       ha='center', va='bottom',
                       fontsize=12, fontweight='bold', color='black')
        plt.show()
    
    def histogram_plot(self, column, data_frame):
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        
        skewness = skew(data_frame[column], nan_policy="omit")
        sns.histplot(data_frame[column], bins=50, kde=True, color="navy", edgecolor="black")
        
        plt.xlabel(column, fontsize=14, fontweight='bold')
        plt.ylabel("Frequency", fontsize=14, fontweight='bold')
        plt.title(f"Distribution of {column}", fontsize=16, fontweight='bold')
        
        plt.text(
            x=data_frame[column].max() * 0.7,
            y=plt.gca().get_ylim()[1] * 0.7,
            s=f"Skewness: {skewness:.2f}",
            fontsize=15, fontweight="bold", color="navy"
        )
        
        plt.show()
    
    def strip_plot(self, column_x, column_y, data_frame):
        plt.figure(figsize=(10, 6))
        sns.stripplot(x=data_frame[column_x], y=data_frame[column_y], jitter=True, alpha=0.7, 
                     palette=["#1f77b4", "#ff7f0e"])
        plt.xlabel(column_x, fontsize=14, fontweight="bold")
        plt.ylabel(column_y, fontsize=14, fontweight="bold")
        plt.title(f"Strip Plot of {column_y} by {column_x}", fontsize=16, fontweight="bold")
        plt.show()
    
    def pie_chart(self, column, data_frame):
        satisfaction_counts = data_frame[column].value_counts()
        
        plt.figure(figsize=(8, 6))
        plt.pie(satisfaction_counts,
                labels=satisfaction_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=plt.cm.Dark2.colors)
        
        plt.title(f"Distribution of {column}")
        plt.axis('equal')
        plt.show()
    
    def box_plot(self, column_x, column_y, data_frame):
        plt.figure(figsize=(10,5))
        sns.boxplot(x=column_x, y=column_y, data=data_frame, palette="rocket")
        
        plt.xlabel(column_x)
        plt.ylabel(column_y)
        plt.title(f'Box Plot of {column_y} by {column_x}')
        plt.show()
    
    def count_plot(self, column, hue, data_frame):
        ax = sns.countplot(x=column, data=data_frame, palette='rocket', hue=hue)
        
        for p in ax.patches:
            try:
                height = p.get_height()
                if height > 0:
                    ax.text(p.get_x() + p.get_width() / 2, height,
                           f'{height/len(data_frame)*100:.2f}%', 
                           ha='center', va='bottom', fontsize=10, color='black')
            except AttributeError:
                continue
        
        plt.title(f'Count of {column} with {hue} Status')
        plt.show()
    
    def scatter_plot(self, x, y, hue, data_frame):
        ax = sns.scatterplot(x=x, y=y, data=data_frame, hue=hue, palette='rocket')
        plt.title(f'Scatter plot of {x} vs {y} by {hue}')
        plt.show()
    
    def heatmap(self, data_frame):
        corr = data_frame.corr()
        ax = sns.heatmap(corr, annot=True, cmap="rocket", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()
    
    def violin_plot(self, x, y, hue, data_frame):
        ax = sns.violinplot(x=x, y=y, data=data_frame, hue=hue, palette='rocket', split=True)
        plt.title(f'Violin plot of {y} across {x} grouped by {hue}')
        plt.show()
    
    def swarm_plot(self, x, y, hue, data_frame):
        sns.swarmplot(x=x, y=y, data=data_frame, hue=hue, palette='rocket', dodge=True)
        plt.title(f'Swarm plot of {y} across {x} grouped by {hue}')
        plt.show()
    
    def kde_plot(self, column, hue, data_frame):
        sns.kdeplot(data=data_frame, x=column, hue=hue, fill=True, palette='rocket')
        plt.title(f'KDE plot of {column} by {hue}')
        plt.show()
    
    def line_plot(self, x, y, hue, data_frame):
        sns.lineplot(x=x, y=y, data=data_frame, hue=hue, palette='rocket')
        plt.title(f'Line plot of {y} vs {x} by {hue}')
        plt.show()
    
    def pixal_bar_plot(self, data_frame, x, label_x, label_y, y='Count'):
        counts = data_frame[x].value_counts().reset_index()
        counts.columns = [x, y]
        
        fig = px.bar(counts, 
                     x=x, 
                     y=y, 
                     title=f"Distribution of {x}",
                     labels={x: label_x, y: label_y},
                     color=y, 
                     color_continuous_scale="ylgnbu")
        
        fig.update_layout(
            title=dict(font=dict(size=24, family="Arial", color="black", weight="bold")),
            xaxis=dict(title_font=dict(size=20, family="Arial", color="black", weight="bold"), 
                       tickfont=dict(size=16, family="Arial", color="black")),
            yaxis=dict(title_font=dict(size=20, family="Arial", color="black", weight="bold"), 
                       tickfont=dict(size=16, family="Arial", color="black")),
            coloraxis_colorbar=dict(title_font=dict(size=16, family="Arial", color="black", weight="bold"))
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        fig.show()
    
    def another_bar_plot(self, data_frame, x, hue, labels_of_hue, y='Count'):
        counts = data_frame.groupby([x, hue]).size().unstack()
        counts.plot(kind='bar', stacked=True, figsize=(14, 6), color=['#1f77b4', '#ff7f0e'], edgecolor="black")
        
        plt.xlabel(x, fontsize=14, labelpad=10, fontweight="bold")
        plt.ylabel(y, fontsize=14, labelpad=10, fontweight="bold")
        plt.title(f"Distribution of {x}", fontsize=16, fontweight="bold")
        
        plt.legend(title=hue, labels=labels_of_hue)
        plt.show()
    
    def bar_pixel_plot(self, data_frame, x, hue):
        grouped = pd.DataFrame(data_frame.groupby(x)[hue].value_counts())
        grouped = grouped.rename(columns={hue: 'Count'})
        grouped = grouped.reset_index()
        
        fig = px.histogram(grouped, x=x, y='count',
                          color=hue, barmode='group', title="<b>"+f"{x} Vs {hue}", text_auto=True)
        
        fig.update_layout(
            title_font_color="black",
            template="plotly",
            title_font_size=30,
            title_x=0.5,
            xaxis_title=x,
            yaxis_title='count')
        
        fig.show()
    
    def for_each_type_bar_pixel_plot(self, data_frame, x, y):
        fig=px.histogram(data_frame, x=x, facet_col=y, color=y, text_auto=True,
                        title="<b>"+f"{y} Vs {x}")
        fig.update_layout(hovermode='x', title_font_size=30)
        
        fig.update_layout(
            title_font_color="black",
            template="plotly",
            title_font_size=30,
            hoverlabel_font_size=20,
            title_x=0.5,
            xaxis_title=x,
            yaxis_title='count')
        
        fig.show()
    
    def grouped_features_relation_with_target_pixel_plot(self, data_frame, x, y, hue):
        fig = px.histogram(
            data_frame,
            x=x,
            y=y,
            orientation="h",
            color=hue,
            text_auto=True,
            title=f"<b>{x} Vs {y}",
            color_discrete_sequence=['#BA1F33', '#3A506B', '#3E885B']
        )
        
        fig.update_layout(
            hovermode='x',
            title_font_size=30,
            title_font_color="black",
            template="plotly",
            title_x=0.5,
            xaxis_title=x,
            yaxis_title=y,
            hoverlabel_font_size=20,
            bargap=0.3
        )
        
        fig.show()
    
    def handle_null_values(self, handling_type, columns, data_frame):
        # handling_type -> mode , mean (not prefered) , median , knn imputer
        
        if handling_type == 'mode':
            for col in columns:
                # replace it with the most frequent value :
                data_frame[col] = data_frame[col].fillna(data_frame[col].mode()[0])
        
        elif handling_type == 'knn imputer':
            for col in columns:
                # replaces it with the previous value, and if it can't find it, then with the next one.
                data_frame[col] = data_frame[col].fillna(method='ffill').fillna(method='bfill')
        
        elif handling_type == 'median':
            for col in columns:
                # replace it with the median because if the data is not normally distributed, the mean will be a problem.
                data_frame[col] = data_frame[col].fillna(data_frame[col].median())
        else:
            print("Invalid Value")
    
    def check_outliers(self, columns, data_frame, whis=1.5):
        fig, axes = plt.subplots(3, 3, figsize=(20, 5 * 3))
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            sns.boxplot(data=data_frame, y=col, ax=axes[i], palette='magma', whis=whis)
            axes[i].set_title(f'Boxplot of {col}', fontsize=12)
            axes[i].set_xlabel('')
            axes[i].set_ylabel(col)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        return fig
    
    def handle_outliers(self, data_frame, column, upper_value=1.5, lower_value=1.5, handle='no'):
        for col in column:
            Q1 = data_frame[col].quantile(0.25)
            Q3 = data_frame[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - (lower_value * IQR)
            upper_bound = Q3 + (upper_value * IQR)
            
            outliers_mask_lower = (data_frame[col] < lower_bound)
            outliers_mask_upper = (data_frame[col] > upper_bound)
            outliers_count_lower = outliers_mask_lower.sum()
            outliers_count_upper = outliers_mask_upper.sum()
            print(f"Number of lower Outliers in {col}: {outliers_count_lower}")
            print(f"Number of upper Outliers in {col}: {outliers_count_upper}")
            
            if handle == 'yes':
                data_frame[col] = np.where(data_frame[col] < lower_bound, lower_bound, data_frame[col])
                data_frame[col] = np.where(data_frame[col] > upper_bound, upper_bound, data_frame[col])
        
        return data_frame
    
    def scaling_data(self, scaler_type, data_frame, features_train, features_test, columns_list):
        if scaler_type == 'standard scaler':
            # Standard Scalar :
            standard_scaler = StandardScaler()
            features_train[columns_list] = standard_scaler.fit_transform(features_train[columns_list])
            features_test[columns_list] = standard_scaler.transform(features_test[columns_list])
        
        elif scaler_type == 'min max scaler':
            # Min Max Scalar :
            min_max_scaler = MinMaxScaler()
            features_train[columns_list] = min_max_scaler.fit_transform(features_train[columns_list])
            features_test[columns_list] = min_max_scaler.transform(features_test[columns_list])
        
        elif scaler_type == 'robust scaler':
            # Robust Scaler :
            robust_scaler = RobustScaler()
            features_train[columns_list] = robust_scaler.fit_transform(features_train[columns_list])
            features_test[columns_list] = robust_scaler.transform(features_test[columns_list])
        else:
            print("There is no scaler type with this name.")
        
        return features_train, features_test
    
    def ordinal_encoding_data(self, features_train, features_test, columns_list):
        # if encoding_type == 'label':
        
        #   label_encoding = LabelEncoder()
        #   features_train=label_encoding.fit_transform(features_train)
        #   features_test=label_encoding.transform(features_test)
        # handling unseen data
        
        all_categories = {}
        for col in columns_list:
            train_cats = features_train[col].unique()
            all_categories[col] = sorted(set(train_cats))
        
        ordinal_encoder = OrdinalEncoder(
            categories=[all_categories[col] for col in columns_list],
            handle_unknown='use_encoded_value',
            unknown_value=len(all_categories[col])
        )
        
        features_train[columns_list] = ordinal_encoder.fit_transform(features_train[columns_list])
        features_test[columns_list] = ordinal_encoder.transform(features_test[columns_list])
        
        return features_train, features_test
    
    def spliting_data(self, data_frame, label, test_size=0.2, random_state=42):
        features = data_frame.drop([label], axis=1)
        target = data_frame[label]
        # random_state = 42 -> to make the split the same every time
        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        return features_train, features_test, target_train, target_test
    
    def correlation(self, features_train, target_train, data_frame, numerical_columns, categorical_columns):
        # Numerical : anova
        x = features_train[numerical_columns]
        y = target_train
        
        f_values, p_values = f_classif(x, y)
        
        numerical_anova_data_frame = pd.DataFrame({
            'Feature': numerical_columns,
            'F-Score': f_values,
            'P-Value': p_values
        }).sort_values(by='F-Score', ascending=False)
        
        # Categorical : chi2
        x = features_train[categorical_columns]
        y = target_train
        
        chi2_selector = SelectKBest(score_func=chi2, k='all')
        chi2_selector.fit(x, y)
        
        categorical_chi2_data_frame = pd.DataFrame({
            'Feature': x.columns,
            'Chi2 Score': chi2_selector.scores_,
            'P-Value': chi2_selector.pvalues_
        }).sort_values(by='Chi2 Score', ascending=False)
        
        return numerical_anova_data_frame, categorical_chi2_data_frame
    
    def best_parameters(self, text, model_type, params, model, x_train, x_test, y_train, y_test):
        result = {}
        
        # grid Search : to find the best hyperparameters
        grid = GridSearchCV(
            # random_state = 42 -> same initial weights for comparsion
            estimator=model,
            # what parameters to try
            param_grid=params,
            # cross validation
            cv=5,
            # get the accuracy
            scoring='accuracy',
            # run on cpu
            n_jobs=-1,
            
            verbose=1
        )
        
        grid.fit(x_train, y_train)
        
        # best_estimator_ : returns the model with the best hyperparameters found during grid search
        best = grid.best_estimator_
        result[f'Best {text} Model'] = best
        
        y_train_pred = best.predict(x_train)
        y_test_pred = best.predict(x_test)
        
        if model_type == 'classification':
            # calculate metrics -> accuracy , precision , recall , f1 , auc
            metrics_train = {
                # accuracy : (TP + TN) / (TP + TN + FP + FN)
                'accuracy': accuracy_score(y_train, y_train_pred),
                # precision : TP / (TP + FP)
                'precision': precision_score(y_train, y_train_pred),
                # recall : TP / (TP + FN)
                'recall': recall_score(y_train, y_train_pred),
                # f1 score : 2 * (precision * recall) / (precision + recall)
                'f1': f1_score(y_train, y_train_pred),
                # auc : area under the roc curve
                'auc': roc_auc_score(y_train, y_train_pred),
                # best parameters from grid search
                'best_params': grid.best_params_
            }
            metrics_test = {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred),
                'recall': recall_score(y_test, y_test_pred),
                'f1': f1_score(y_test, y_test_pred),
                'auc': roc_auc_score(y_test, y_test_pred),
                'best_params': grid.best_params_
            }
            print(f"Best {text}")
            print(f"Best Parameters: {grid.best_params_}")
            print("Accuracies are based on Accuracy Score:")
            print(f"Train Accuracy: {metrics_train['accuracy']:.4f}")
            print(f"Test Accuracy: {metrics_test['accuracy']:.4f}")
        
        elif model_type == 'regression':
            metrics_train = {
                # R² : proportion of variance explained by the model
                'r2': r2_score(y_train, y_train_pred),
                # MAE : mean absolute error
                'mae': mean_absolute_error(y_train, y_train_pred),
                # MSE : mean squared error
                'mse': mean_squared_error(y_train, y_train_pred),
                # RMSE : root mean squared error
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                # best parameters from grid search
                'best_params': grid.best_params_
            }
            metrics_test = {
                'r2': r2_score(y_test, y_test_pred),
                'mae': mean_absolute_error(y_test, y_test_pred),
                'mse': mean_squared_error(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'best_params': grid.best_params_
            }
            print(f"Best {text}")
            print(f"Best Parameters: {grid.best_params_}")
            print("Accuracies are based on R² Score:")
            print(f"Train Accuracy: {metrics_train['r2']:.4f}")
            print(f"Test Accuracy: {metrics_test['r2']:.4f}")
        
        elif model_type == 'unsupervised':
            # calculate metrics -> silhouette , davies-bouldin , calinski-harabasz
            metrics_train = {
                # silhouette : cohesion vs separation (-1 to 1, higher is better)
                'silhouette': silhouette_score(x_train, y_train_pred),
                # davies-bouldin : average similarity between clusters (lower is better)
                'davies_bouldin': davies_bouldin_score(x_train, y_train_pred),
                # calinski-harabasz : ratio of between-cluster dispersion to within-cluster dispersion (higher is better)
                'calinski_harabasz': calinski_harabasz_score(x_train, y_train_pred),
                # best parameters from grid search
                'best_params': grid.best_params_
            }
            metrics_test = {
                'silhouette': silhouette_score(x_test, y_test_pred),
                'davies_bouldin': davies_bouldin_score(x_test, y_test_pred),
                'calinski_harabasz': calinski_harabasz_score(x_test, y_test_pred),
                'best_params': grid.best_params_
            }
            print(f"Best {text}")
            print(f"Best Parameters: {grid.best_params_}")
            print("Accuracies are based on Silhouette Score:")
            print(f"Train Accuracy: {metrics_train['silhouette']:.4f}")
            print(f"Test Accuracy: {metrics_test['silhouette']:.4f}")
        
        result[f'{text} Train Metrics'] = metrics_train
        result[f'{text} Test Metrics'] = metrics_test
        
        return result
    
    def train_test_evaluate(self, scaler_type, x_train, x_test, y_train, y_test, model):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred_train = model.predict(x_train)
        print(f"{scaler_type} Scaler")
        print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
        print("Test Accuracy:", accuracy_score(y_test, y_pred))
    
    def evaluate_classification(self, y_test, y_pred, labels=None):
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("
Classification Report:
", classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    
    def evaluate_regression(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
    
    def evaluate_clustering(self, X, labels, true_labels=None):
        # Internal metrics (no ground truth needed)
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        
        print(f"Silhouette Score: {sil:.4f}")
        print(f"Davies-Bouldin Index: {db:.4f}")
        
        # External metric (only if true labels are available)
        if true_labels is not None:
            ari = adjusted_rand_score(true_labels, labels)
            print(f"Adjusted Rand Index (vs true labels): {ari:.4f}")
