import pandas as pd
import math
import re
from collections import Counter

from config import NGRAM_SIZE, NGRAM_TOP_K, FEATURE_CONFIG

class DomainFeatureExtractor:
    def __init__(self):
        self.common_ngrams = set()
        self.ngram_freq = Counter()
        
    def train_ngram_model(self, domains):
        """
        Learns the most common n-grams from benign domains
        
        Parameters:
        domains (list): List of benign domains for training
        
        Returns:
        set: Set of common n-grams
        """
        ngram_counter = Counter()
        
        # Extract n-grams from all benign domains
        for domain in domains:
            domain_parts = domain.split('.')
            if len(domain_parts) > 1:
                domain_no_tld = '.'.join(domain_parts[:-1])
            else:
                domain_no_tld = domain_parts[0]
                
            # Extract n-grams
            ngrams = [domain_no_tld[i:i+NGRAM_SIZE] for i in range(len(domain_no_tld) - NGRAM_SIZE + 1)]
            
            ngram_counter.update(ngrams)
        
        # Select the top_k most frequent n-grams
        self.ngram_freq = ngram_counter
        self.common_ngrams = set([ngram for ngram, _ in ngram_counter.most_common(NGRAM_TOP_K)])
        
        
        print(f"N-gram model trained: identified {len(self.common_ngrams)} common n-grams")
        return self.common_ngrams
        
    def extract_features(self, domain):
        """Extract lexical features from a domain name"""
        features = {}
        
        # Domain length
        if FEATURE_CONFIG['length']:
            features['length'] = len(domain)
        
        # Proportion of alphanumeric characters
        if FEATURE_CONFIG['alphanumeric_ratio']:
            alphanumeric_chars = sum(c.isalnum() for c in domain)
            features['alphanumeric_ratio'] = alphanumeric_chars / len(domain) if len(domain) > 0 else 0
        
        # Entropy of domain name - measures randomness of characters
        # Higher entropy often indicates algorithmically generated domains
        if FEATURE_CONFIG['entropy']:
            features['entropy'] = self._calculate_entropy(domain)
        
        # N-gram features
        if FEATURE_CONFIG['ngram_features']:
            ngram_features = self._extract_ngram_features(domain)
            features.update(ngram_features)
        
        return features
    
    def _calculate_entropy(self, string):
        """Calculate Shannon entropy of a string"""
        if not string:
            return 0
        
        # Count character frequencies
        counter = Counter(string)
        total_chars = len(string)
        
        # Calculate entropy
        entropy = 0
        for count in counter.values():
            probability = count / total_chars
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _extract_ngram_features(self, domain):
        """Extract n-gram features from domain name"""
        ngram_features = {}
        
        # Use default list if not trained
        if not self.common_ngrams:
            raise ValueError("N-gram model not trained.")
        
        # Remove TLD for more accurate analysis
        domain_parts = domain.split('.')
        if len(domain_parts) > 1:
            domain_no_tld = '.'.join(domain_parts[:-1])
        else:
            domain_no_tld = domain_parts[0]
        
        # Extract n-grams (substrings of length n)
        ngrams = [domain_no_tld[i:i+NGRAM_SIZE] for i in range(len(domain_no_tld) - NGRAM_SIZE + 1)]
        
        # Count rare n-grams (those not in common_ngrams)
        # DGA domains typically have more rare n-grams
        rare_ngrams = [ng for ng in ngrams if ng not in self.common_ngrams]
        
        # Calculate features
        ngram_features['rare_ngram_ratio'] = len(rare_ngrams) / len(ngrams) if ngrams else 0
        ngram_features['rare_ngram_count'] = len(rare_ngrams)
        
        # Calculate average frequency of n-grams in this domain compared to training data
        avg_freq = sum(self.ngram_freq.get(ng, 0) for ng in ngrams) / len(ngrams) if ngrams else 0
        ngram_features['ngram_avg_freq'] = avg_freq
        
        # Calculate rarity score (higher means more rare n-grams)
        max_freq = max(self.ngram_freq.values()) if self.ngram_freq else 1
        rarity_score = 1.0 - (avg_freq / max_freq)
        ngram_features['ngram_rarity_score'] = rarity_score
        
        return ngram_features
    
    def preprocess_domains(self, domains):
        """Extract features from a list of domains"""
        feature_list = []
        
        for domain in domains:
            features = self.extract_features(domain)
            feature_list.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_list)
        
        return df


class DNSLogParser:
    def parse_dnsmasq_log(self, log_file):
        """Parse DNS logs from dnsmasq"""
        domains = []
        timestamps = []
        
        with open(log_file, 'r') as f:
            for line in f:
                # Extract domain name and timestamp
                match = re.search(r'(\w+\s+\d+\s+\d+:\d+:\d+).*query\[\w+\]\s+([a-zA-Z0-9.-]+)', line)
                if match:
                    timestamp, domain = match.groups()
                    domains.append(domain)
                    timestamps.append(timestamp)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps, format='%b %d %H:%M:%S', errors='coerce'),
            'domain': domains
        })
        
        return df