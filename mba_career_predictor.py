import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
# IMPORTANT: This must match the name of your CSV data file
FILE_PATH = "data.csv"
RANDOM_SEED = 42

# Define the function to simplify the target variable (Industry)
def simplify_target_industry(industry):
    """Groups 11 specific industries into 4 broad, balanced sectors."""
    # Grouping based on typical MBA job tracks
    tech_digital = ['Technology', 'Marketing/Advertising', 'Arts & Entertainment']
    finance_consulting = ['Finance', 'Consulting']
    operations_manufacturing = ['Manufacturing', 'Retail', 'Logistics']
    public_social = ['Education', 'Healthcare', 'Government', 'Other']

    if industry in tech_digital:
        return '1_Tech_Digital'
    elif industry in finance_consulting:
        return '2_Finance_Consulting'
    elif industry in operations_manufacturing:
        return '3_Operations_Manufacturing'
    elif industry in public_social:
        return '4_Public_Social'
    else:
        # Fallback for any unknown industry
        return '5_Unknown'

# --- 2. DATA LOADING AND INITIAL CLEANING ---

def load_and_preprocess_data(file_path):
    """Loads, cleans, and transforms the raw data."""
    print("--- Loading and Cleaning Data ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None

    # Rename columns for simpler access
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    df.rename(columns={'Timestamp': 'Timestamp', 
                       'Inwhichindustryareyoupresentlyworking': 'Target_Industry'}, inplace=True)

    # Apply target simplification
    df['Target_Sector'] = df['Target_Industry'].apply(simplify_target_industry)
    print("New Target Distribution (Sectors):")
    print(df['Target_Sector'].value_counts())

    # Define features and target
    y = df['Target_Sector']
    
    # Drop irrelevant columns and the original specific industry column
    features_to_drop = [
        'Timestamp', 
        'Target_Industry', 
        'Target_Sector',
        'Pleasespecifyyourmajorfieldofstudyifapplicable', 
        'Howwouldyouliketowork', 
        'Howdoyoudefinecareersuccess', 
        'Pleaseprovideanyadditionalcommentsthatmighthelpusunderstandyourcareeraspirations', 
        'SpecifyindustryifchoosingOtherskipifindustrypreviouslychosen'
    ]
    X = df.drop(columns=features_to_drop, errors='ignore')

    # Handle Multi-Choice Skill Column ('WhatareyourtopthreeskillsSelectthree')
    skills_col = 'WhatareyourtopthreeskillsSelectthree'
    if skills_col in X.columns:
        skills_df = X[skills_col].str.get_dummies(sep='; ')
        X = pd.concat([X, skills_df], axis=1)
        X.drop(columns=[skills_col], inplace=True)

    # Handle missing values by filling them with the mode
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype == 'object':
                X[col].fillna(X[col].mode()[0], inplace=True)
            else:
                X[col].fillna(X[col].mode()[0], inplace=True) # Use mode for numeric as well

    # Encode the Target Variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Encode Features (X) using One-Hot Encoding
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    return X_encoded, y_encoded, le


# --- 3. MODEL TRAINING AND EVALUATION ---

def train_and_evaluate_model(X, y, label_encoder):
    """Trains the Random Forest model and prints evaluation metrics."""
    print("\n--- Training Random Forest Model ---")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    # Initialize and Train the Random Forest Classifier
    # class_weight='balanced' helps with the slight remaining imbalance
    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=RANDOM_SEED, class_weight='balanced')
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)

    print(f"\n## Model Performance (4 Sectors)")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)

    # Plot feature importance
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top_10_features = feature_importances.sort_values(ascending=False).nlargest(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_10_features.values, y=top_10_features.index, palette="viridis")
    plt.title('Top 10 Feature Importances for Career Sector Prediction')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig("feature_importance_plot_4_sectors.png")
    print("\n[Saved Feature Importance Plot as feature_importance_plot_4_sectors.png]")

    return model, X_train.columns

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # Set plotting style
    sns.set_style("whitegrid")
    
    X_encoded, y_encoded, le = load_and_preprocess_data(FILE_PATH)
    
    if X_encoded is not None:
        train_and_evaluate_model(X_encoded, y_encoded, le)