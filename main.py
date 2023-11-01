import pandas as pd
from sklearn.metrics import accuracy_score
from utils import load_data, split_data
from data_generators import gaussian_generator, binomial_generator, uniform_generator
from probability_functions import pdf_uniform, pdf_gaussian, pmf_binomial
from data_analysis import compute_training_params, compute_breed_proportions, process_email
from naive_bayes import naive_bayes_classifier, word_freq_per_class, class_frequencies

# Load and prepare datasets
df_all_breeds, emails = load_data()

# Generate breed data if needed
try:
    df_all_breeds_generated = generate_data(gaussian_generator, binomial_generator, uniform_generator)
    if not df_all_breeds_generated.equals(df_all_breeds):
        raise ValueError("Generated dataset does not match the pre-loaded one.")
except Exception as e:
    print(f"Error in data generation: {e}. Using pre-loaded dataset.")
    df_all_breeds_generated = df_all_breeds

# Split data into training and testing
df_train, df_test = split_data(df_all_breeds_generated)

# Compute parameters for training data
train_params = compute_training_params(df_train)

# Preprocess emails dataset
emails['processed_text'] = emails['text'].apply(process_email)
word_freq = word_freq_per_class(emails)
class_freq = class_frequencies(emails)

# Naive Bayes Classification
test_emails = ["Sample email text 1", "Sample email text 2", ...]  # Sample test emails
for email in test_emails:
    prob_spam = naive_bayes_classifier(email, word_freq, class_freq)
    print(f"Email: {email}\nProbability of being spam: {prob_spam:.2f}\n")


# Predict and Evaluate Breed Classification
from data_analysis import predict_breed

# Function to predict breed for each dog in the test set
def predict_breeds_for_test(df_test, train_params, class_probs):
    return df_test.apply(lambda x: predict_breed([*x[FEATURES]], FEATURES, train_params, class_probs), axis=1)

# Predict breeds for the test dataset
predicted_breeds = predict_breeds_for_test(df_test, train_params, train_class_probs)

# Calculate and print accuracy
test_accuracy = accuracy_score(df_test["breed"], predicted_breeds)
print(f"Accuracy of breed classification: {test_accuracy:.2f}")

# Evaluate Naive Bayes Classifier Accuracy on Email Dataset
def evaluate_naive_bayes(emails, word_freq, class_freq):
    correct_predictions = 0

    for _, row in emails.iterrows():
        text = row['text']
        true_class = row['spam']
        predicted_prob_spam = naive_bayes_classifier(text, word_freq, class_freq)
        predicted_class = 1 if predicted_prob_spam >= 0.5 else 0

        if predicted_class == true_class:
            correct_predictions += 1

    return correct_predictions / len(emails)

# Calculate accuracy
email_accuracy = evaluate_naive_bayes(emails, word_freq, class_freq)
print(f"Accuracy of Naive Bayes classifier on email dataset: {email_accuracy:.2f}")
