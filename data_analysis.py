import pandas as pd
import numpy as np
from data_generators import gaussian_generator, binomial_generator, uniform_generator

# Load pre-existing data
try:
    pre_loaded_df = pd.read_pickle("df_all_breeds.pkl")
except FileNotFoundError:
    print("File not found. Ensure 'df_all_breeds.pkl' is in the correct directory.")

# Read the pre-loaded dataset
FEATURES = ["height", "weight", "bark_days", "ear_head_ratio"]

def generate_and_validate_data():
    try:
        df_all_breeds = generate_data(gaussian_generator, binomial_generator, uniform_generator)
    except Exception as e:
        print(f"There was an error when generating the dataset: {e}")
        df_all_breeds = pre_loaded_df
    else:
        if not df_all_breeds.equals(pre_loaded_df):
            print("Generated dataset differs from pre-loaded one. Using pre-loaded dataset.")
            df_all_breeds = pre_loaded_df
    return df_all_breeds

def split_data(df_all_breeds):
    split = int(len(df_all_breeds) * 0.7)
    df_train = df_all_breeds[:split].reset_index(drop=True)
    df_test = df_all_breeds[split:].reset_index(drop=True)
    return df_train, df_test

def compute_training_params(df_train, features):
    """
    Computes the estimated distribution parameters for each feature of each breed.

    Args:
        df_train (DataFrame): The training dataset containing features and breed information.
        features (list): List of feature names to calculate parameters for.

    Returns:
        dict: A nested dictionary where each key is a breed and each value is another dictionary.
              This inner dictionary's keys are feature names and values are the estimated parameters.
    """
    params_dict = {}
    
    for breed in df_train['breed'].unique():
        breed_data = df_train[df_train['breed'] == breed]
        feature_params = {}

        for feature in features:
            if feature in ["height", "weight"]:  # Assuming Gaussian distribution
                mu = breed_data[feature].mean()
                sigma = breed_data[feature].std()
                feature_params[feature] = {'mu': mu, 'sigma': sigma}
            
            elif feature == "bark_days":  # Assuming Binomial distribution
                # Example: total number of days considered could be a constant, say 30
                # n is then 30, and p is the mean of bark_days / n
                n = 30  # This can be adjusted based on the actual definition of 'bark_days'
                p = breed_data[feature].mean() / n
                feature_params[feature] = {'n': n, 'p': p}

            elif feature == "ear_head_ratio":  # Assuming Uniform distribution
                a = breed_data[feature].min()
                b = breed_data[feature].max()
                feature_params[feature] = {'a': a, 'b': b}

        params_dict[breed] = feature_params

    return params_dict


def compute_breed_proportions(df):
    """
    Computes the estimated probabilities of each breed.
    """
    probs_dict = {}
    for breed in df['breed'].unique():
        df_breed = df[df["breed"] == breed]
        prob_class = len(df_breed) / len(df)
        probs_dict[breed] = round(prob_class, 3)
    return probs_dict

# Example usage
df_all_breeds = generate_and_validate_data()
df_train, df_test = split_data(df_all_breeds)
train_params = compute_training_params(df_train, FEATURES)
train_class_probs = compute_breed_proportions(df_train)
