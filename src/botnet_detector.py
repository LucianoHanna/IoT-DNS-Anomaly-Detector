import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from feature_extraction import DomainFeatureExtractor
from config import SVM_PARAMS

class BotnetDetector:
    def __init__(self, model_dir='models'):
        """Initialize the botnet detector"""
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_extractor = DomainFeatureExtractor()
        
        # Extract experiment ID from model_dir path
        self.exp_id = os.path.basename(model_dir)
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def train(self, domains):
        """Train the One-class SVM model on benign domains"""
        print(f"Extracting features from {len(domains)} domains...")
        
        # Train the n-gram model to identify common patterns
        print("Training n-gram model on benign domains...")
        self.feature_extractor.train_ngram_model(domains)
        
        # Extract features from benign domains
        features_df = self.feature_extractor.preprocess_domains(domains)
        
        # Save the n-gram model
        self._save_ngram_model()
        
        # Standardize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features_df)
        
        # Initialize One-class SVM with parameters from config
        print("Training One-class SVM model...")
        self.model = svm.OneClassSVM(
            nu=SVM_PARAMS['nu'],
            kernel=SVM_PARAMS['kernel'],
            gamma=SVM_PARAMS['gamma']
        )
        self.model.fit(features_scaled)
        
        # Save model and scaler
        self._save_model()
        
        return self.model
    
    def _save_model(self):
        """Save model and scaler to disk"""
        if self.model is not None and self.scaler is not None:
            joblib.dump(self.model, os.path.join(self.model_dir, f'oneclass_svm_model_{self.exp_id}.pkl'))
            joblib.dump(self.scaler, os.path.join(self.model_dir, f'scaler_{self.exp_id}.pkl'))
            print(f"Model and scaler saved to {self.model_dir}")
    
    def _save_ngram_model(self):
        """Save n-gram model to disk"""
        ngram_model = {
            'common_ngrams': self.feature_extractor.common_ngrams,
            'ngram_freq': self.feature_extractor.ngram_freq
        }
        joblib.dump(ngram_model, os.path.join(self.model_dir, f'ngram_model_{self.exp_id}.pkl'))
        print(f"N-gram model saved to {self.model_dir}")
    
    def _load_ngram_model(self):
        """Load n-gram model from disk"""
        ngram_model_path = os.path.join(self.model_dir, f'ngram_model_{self.exp_id}.pkl')
        if os.path.exists(ngram_model_path):
            ngram_model = joblib.load(ngram_model_path)
            self.feature_extractor.common_ngrams = ngram_model['common_ngrams']
            self.feature_extractor.ngram_freq = ngram_model['ngram_freq']
            self.feature_extractor.trained = True
            print(f"N-gram model loaded from {self.model_dir}")
            return True
        else:
            print(f"N-gram model not found in {self.model_dir}")
            return False
    
    def load_model(self):
        """Load model and scaler from disk"""
        model_path = os.path.join(self.model_dir, f'oneclass_svm_model_{self.exp_id}.pkl')
        scaler_path = os.path.join(self.model_dir, f'scaler_{self.exp_id}.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Carregar também o modelo de n-gramas
            self._load_ngram_model()
            
            print(f"Model and scaler loaded from {self.model_dir}")
            return True
        else:
            print(f"Model or scaler not found in {self.model_dir}")
            return False
    
    def predict(self, domains):
        """Predict if domains are anomalous (potential botnet)"""
        if self.model is None or self.scaler is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded")
        
        # Extract features
        features_df = self.feature_extractor.preprocess_domains(domains)
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Predict anomalies
        predictions = self.model.predict(features_scaled)
        
        # Convert to binary: -1 (outlier) -> 1 (anomaly), 1 (inlier) -> 0 (normal)
        predictions_binary = np.where(predictions == -1, 1, 0)
        
        # Calculate anomaly score from decision function
        # More negative values indicate stronger anomalies
        decision_scores = self.model.decision_function(features_scaled)
        anomaly_scores = -decision_scores  # Invert so higher = more anomalous
        
        # Normalize scores to [0,1] range
        min_score = np.min(anomaly_scores)
        max_score = np.max(anomaly_scores)
        if max_score > min_score:
            normalized_scores = (anomaly_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(anomaly_scores)
        
        results = pd.DataFrame({
            'domain': domains,
            'is_anomaly': predictions_binary,
            'anomaly_score': normalized_scores,
            'rare_ngram_ratio': features_df['rare_ngram_ratio'] if 'rare_ngram_ratio' in features_df.columns else 0,
            'entropy': features_df['entropy'] if 'entropy' in features_df.columns else 0
        })
        
        return results
    
    def evaluate(self, benign_domains, malicious_domains):
        """Evaluate model performance"""
        if self.model is None or self.scaler is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded")
        
        # Extract and scale features for benign domains
        benign_features_df = self.feature_extractor.preprocess_domains(benign_domains)
        benign_features_scaled = self.scaler.transform(benign_features_df)
        
        # Extract and scale features for malicious domains
        malicious_features_df = self.feature_extractor.preprocess_domains(malicious_domains)
        malicious_features_scaled = self.scaler.transform(malicious_features_df)
        
        # Predict on benign domains
        benign_predictions = self.model.predict(benign_features_scaled)
        benign_predictions_binary = np.where(benign_predictions == -1, 1, 0)
        
        # Predict on malicious domains
        malicious_predictions = self.model.predict(malicious_features_scaled)
        malicious_predictions_binary = np.where(malicious_predictions == -1, 1, 0)
        
        # Create true labels
        y_true_benign = np.zeros_like(benign_predictions_binary)  # 0 = normal
        y_true_malicious = np.ones_like(malicious_predictions_binary)  # 1 = anomaly
        
        # Combine predictions and true labels
        y_pred = np.concatenate([benign_predictions_binary, malicious_predictions_binary])
        y_true = np.concatenate([y_true_benign, y_true_malicious])
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate specific metrics
        tn, fp, fn, tp = cm.ravel()
        
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Analyze feature importance
        feature_importance = self._analyze_feature_importance(benign_features_df, malicious_features_df)
        
        # Return results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'false_positive_rate': false_positive_rate,
            'detection_rate': detection_rate,
            'feature_importance': feature_importance
        }
        
        return results
    
    def _analyze_feature_importance(self, benign_df, malicious_df):
        """Analyze which features are most discriminative between benign and malicious domains"""
        importance = {}
        
        # Calculate mean differences for each feature
        for col in benign_df.columns:
            benign_mean = benign_df[col].mean()
            malicious_mean = malicious_df[col].mean()
            
            # Normalize by standard deviation
            std = np.std(np.concatenate([benign_df[col], malicious_df[col]]))
            if std > 0:
                importance[col] = abs(benign_mean - malicious_mean) / std
            else:
                importance[col] = 0
        
        return importance
    
    def evaluate_dga_families(self, dga_families):
        """Evaluate model performance on different DGA families"""
        if self.model is None or self.scaler is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded")
        
        family_results = {}
        all_feature_dfs = {}
        
        for family, domains in dga_families.items():
            # Extract and scale features
            features_df = self.feature_extractor.preprocess_domains(domains)
            features_scaled = self.scaler.transform(features_df)
            
            # Predict
            predictions = self.model.predict(features_scaled)
            predictions_binary = np.where(predictions == -1, 1, 0)
            
            # Calculate decision scores
            decision_scores = self.model.decision_function(features_scaled)
            
            # Calculate detection rate and other metrics
            detection_rate = np.mean(predictions_binary)
            avg_decision_score = np.mean(-decision_scores)  # Negative so higher = more anomalous
            
            family_results[family] = {
                'detection_rate': detection_rate,
                'avg_anomaly_score': avg_decision_score,
                'num_domains': len(domains)
            }
            
            # Store feature data for analysis
            all_feature_dfs[family] = features_df
        
        # Calculate feature importance per family
        feature_importance_by_family = {}
        benign_features = self.feature_extractor.preprocess_domains(dga_families[list(dga_families.keys())[0]][:1])
        benign_columns = benign_features.columns
        
        for family, features_df in all_feature_dfs.items():
            # Analyze which features are most distinctive for this family
            importance = {}
            for col in benign_columns:
                if col in features_df.columns:
                    importance[col] = features_df[col].mean()
            
            feature_importance_by_family[family] = importance
        
        return family_results, feature_importance_by_family
