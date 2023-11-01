
# Dog Breed Classification and Email Spam Detection using Naive Bayes

This project applies the Naive Bayes algorithm to two different tasks: classifying dog breeds based on various features, and detecting spam emails. The project structure and code are designed to demonstrate the versatility of Naive Bayes in handling distinct types of data and classification challenges.

## Project Structure

The project is organized into several Python scripts, each handling different aspects of the project:

1. `data_generators.py`: Contains functions to generate synthetic data for dog breeds using different statistical distributions.

2. `probability_functions.py`: Includes probability density functions and mass functions used in the Naive Bayes classifier.

3. `data_analysis.py`: Provides tools for analyzing and preparing the data for model training and testing.

4. `naive_bayes.py`: Implements the Naive Bayes classifier for both the dog breed classification and the email spam detection tasks.

5. `main.py`: The main script that uses functions from other modules to perform data loading, preprocessing, model training, prediction, and evaluation.

6. `utils.py`: A utility module containing functions for data loading, preprocessing, and other common tasks.

## Setup and Installation

Ensure you have Python 3.x installed on your machine. Clone this repository, and install the required packages:

```bash
git clone <repository_url>
cd <repository_directory>
pip install -r requirements.txt
```

## Running the Project

To run the project, simply execute the `main.py` script:

```bash
python main.py
```

## Features and Implementations

- **Dog Breed Classification**: Utilizes synthetic data generated based on predefined parameters for different dog breeds. The Naive Bayes classifier is trained to classify breeds based on features like height, weight, etc.

- **Email Spam Detection**: Applies the Naive Bayes classifier to distinguish between spam and non-spam (ham) emails. The classifier is trained and tested on a dataset of emails.


