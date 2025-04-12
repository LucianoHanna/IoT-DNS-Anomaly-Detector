"""
Configuration file for IoT Botnet Detection System
Contains all tunable parameters for the system
"""

# Data splitting parameters
TRAIN_TEST_SPLIT = 0.8  # Percentage of data used for training (0.8 = 80%)

# N-gram model parameters
NGRAM_SIZE = 2          # Size of n-grams (bigrams by default)
NGRAM_TOP_K = 75        # Number of most common n-grams to consider as "normal"

# Feature extraction parameters
# You can enable/disable specific features or add weights if needed
FEATURE_CONFIG = {
    'length': True,                 # Use domain length as a feature
    'alphanumeric_ratio': True,     # Use proportion of alphanumeric characters
    'entropy': True,                # Use character entropy
    'ngram_features': True,         # Use n-gram based features
    'vowel_ratio': True,           # Ratio of vowels in the domain
    'digit_ratio': False,           # Ratio of digits in the domain
    'subdomain_count': True,        # Subdomain count
}

# One-class SVM model parameters
SVM_PARAMS = {
    'nu': 0.1,                  # Upper bound on training errors / Lower bound on support vectors
    'kernel': 'rbf',            # Kernel type: 'rbf', 'linear', 'poly', 'sigmoid'
    'gamma': 'scale',           # Kernel coefficient: 'scale', 'auto', or float value
}
