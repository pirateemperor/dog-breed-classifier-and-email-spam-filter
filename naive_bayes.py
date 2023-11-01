def naive_bayes_classifier(text, word_freq, class_freq):
    """
    Implements a naive Bayes classifier to determine the probability of an email being spam.

    Args:
        text (str): The input email text to classify.
        word_freq (dict): A dictionary containing word frequencies in the training corpus. 
        The keys are words, and the values are dictionaries containing frequencies for 'spam' and 'ham' classes.
        class_freq (dict): A dictionary containing class frequencies in the training corpus. 
        The keys are class labels ('spam' and 'ham'), and the values are the respective frequencies.

    Returns:
        float: The probability of the email being spam.
    """
    text = text.lower()
    words = set(text.split())
    cumulative_product_spam = 1.0
    cumulative_product_ham = 1.0

    # Calculate likelihoods
    for word in words:
        if word in word_freq:
            # Calculate the likelihood of this word appearing in a spam email
            spam_likelihood = (word_freq[word]['spam'] + 1) / (class_freq['spam'] + 2)  # Laplace smoothing
            # Calculate the likelihood of this word appearing in a ham email
            ham_likelihood = (word_freq[word]['ham'] + 1) / (class_freq['ham'] + 2)    # Laplace smoothing
            
            cumulative_product_spam *= spam_likelihood
            cumulative_product_ham *= ham_likelihood

    # Prior probabilities
    prior_spam = class_freq['spam'] / (class_freq['spam'] + class_freq['ham'])
    prior_ham = class_freq['ham'] / (class_freq['spam'] + class_freq['ham'])

    # Posterior probabilities
    posterior_spam = cumulative_product_spam * prior_spam
    posterior_ham = cumulative_product_ham * prior_ham

    # Normalize to get probabilities
    prob_spam = posterior_spam / (posterior_spam + posterior_ham)

    return prob_spam
