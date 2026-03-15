"""
SMS Spam Classifier - Google Colab Version

A complete machine learning project to classify SMS messages as spam or ham.
Optimized for Google Colab with interactive widgets and visualizations.

Usage in Colab:
1. Upload spam.csv file
2. Run all cells
3. Use the interactive widget to test predictions
"""

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Colab-specific imports
try:
    from google.colab import files
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("Not running in Colab - some features may not work")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


class SpamClassifierColab:
    """Spam classifier optimized for Google Colab"""
    
    def __init__(self):
        """Initialize the spam classifier"""
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
        print("=" * 70)
        print("🤖 SMS SPAM CLASSIFIER - GOOGLE COLAB EDITION")
        print("=" * 70)
    
    def upload_dataset(self):
        """Upload dataset in Colab"""
        if IN_COLAB:
            print("\n📂 Please upload your spam.csv file:")
            uploaded = files.upload()
            
            # Get the uploaded filename
            filename = list(uploaded.keys())[0]
            return filename
        else:
            return 'spam.csv'
    
    def load_data(self, filepath='spam.csv'):
        """Load and clean the dataset"""
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)
        
        # Load data
        self.df = pd.read_csv(filepath, encoding='latin-1')
        
        # Keep only relevant columns
        self.df = self.df[['v1', 'v2']]
        self.df.columns = ['label', 'message']
        
        # Remove duplicates
        original_size = len(self.df)
        self.df = self.df.drop_duplicates(subset='message', keep='first')
        duplicates_removed = original_size - len(self.df)
        
        print(f"✓ Loaded {len(self.df)} messages")
        print(f"✓ Removed {duplicates_removed} duplicates")
        
        # Display class distribution
        class_dist = self.df['label'].value_counts()
        print(f"\n📊 Class Distribution:")
        print(class_dist)
        print(f"\n📈 Spam percentage: {(self.df['label']=='spam').sum()/len(self.df)*100:.2f}%")
        
        # Show sample messages
        print("\n📝 Sample Messages:")
        print("\nHam (Legitimate):")
        print(self.df[self.df['label']=='ham']['message'].iloc[0])
        print("\nSpam:")
        print(self.df[self.df['label']=='spam']['message'].iloc[0])
        
        return self.df
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\d{10}', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def feature_engineering(self):
        """Create additional features"""
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING")
        print("=" * 70)
        
        self.df['message_length'] = self.df['message'].apply(len)
        self.df['num_words'] = self.df['message'].apply(lambda x: len(x.split()))
        self.df['num_special_chars'] = self.df['message'].apply(
            lambda x: len([c for c in x if c in string.punctuation])
        )
        self.df['num_uppercase'] = self.df['message'].apply(
            lambda x: len([c for c in x if c.isupper()])
        )
        self.df['num_digits'] = self.df['message'].apply(
            lambda x: len([c for c in x if c.isdigit()])
        )
        
        # Clean messages
        print("🧹 Preprocessing text...")
        self.df['cleaned_message'] = self.df['message'].apply(self.preprocess_text)
        
        print("✓ Created additional features")
        print("✓ Preprocessed text")
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("\n" + "=" * 70)
        print("PREPARING DATA")
        print("=" * 70)
        
        X = self.df['cleaned_message']
        y = self.df['label'].map({'ham': 0, 'spam': 1})
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✓ Training set: {len(self.X_train)} samples")
        print(f"✓ Test set: {len(self.X_test)} samples")
        print(f"✓ Train spam ratio: {self.y_train.sum()/len(self.y_train)*100:.2f}%")
        print(f"✓ Test spam ratio: {self.y_test.sum()/len(self.y_test)*100:.2f}%")
    
    def train_models(self):
        """Train multiple models and compare performance"""
        print("\n" + "=" * 70)
        print("TRAINING MODELS")
        print("=" * 70)
        
        # Define models
        models_to_train = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='linear', probability=True, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        # Vectorize text using TF-IDF
        print("\n1️⃣ Vectorizing text using TF-IDF...")
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        X_train_vec = self.vectorizer.fit_transform(self.X_train)
        X_test_vec = self.vectorizer.transform(self.X_test)
        
        print(f"✓ Text vectorization complete")
        print(f"  📐 Feature dimension: {X_train_vec.shape[1]}")
        
        # Train and evaluate each model
        results = []
        
        print("\n2️⃣ Training models...")
        for name, model in models_to_train.items():
            print(f"\n   🔄 Training {name}...")
            
            # Train
            model.fit(X_train_vec, self.y_train)
            
            # Predict
            y_pred = model.predict(X_test_vec)
            
            # Evaluate
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Store model and results
            self.models[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred
            }
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            
            print(f"   ✓ Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
                  f"Recall: {recall:.4f} | F1: {f1:.4f}")
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Find best model
        best_idx = results_df['F1-Score'].idxmax()
        self.best_model_name = results_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]['model']
        
        print("\n" + "=" * 70)
        print("📊 MODEL COMPARISON")
        print("=" * 70)
        print(results_df.to_string(index=False))
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print("=" * 70)
        
        return results_df
    
    def evaluate_best_model(self):
        """Detailed evaluation of the best model"""
        print("\n" + "=" * 70)
        print(f"📈 DETAILED EVALUATION - {self.best_model_name}")
        print("=" * 70)
        
        y_pred = self.models[self.best_model_name]['predictions']
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Ham', 'Spam']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\n🔢 Confusion Matrix:")
        print(f"                Predicted")
        print(f"              Ham    Spam")
        print(f"Actual Ham    {cm[0][0]:<6} {cm[0][1]:<6}")
        print(f"       Spam   {cm[1][0]:<6} {cm[1][1]:<6}")
        
        return cm
    
    def plot_results(self, results_df, confusion_matrix):
        """Create comprehensive visualizations"""
        print("\n" + "=" * 70)
        print("🎨 CREATING VISUALIZATIONS")
        print("=" * 70)
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Model Comparison
        ax1 = plt.subplot(2, 3, 1)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(results_df))
        width = 0.2
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        for i, metric in enumerate(metrics):
            ax1.bar(x + i*width, results_df[metric], width, label=metric, color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('📊 Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        # 2. Best Model Metrics
        ax2 = plt.subplot(2, 3, 2)
        best_metrics = results_df[results_df['Model'] == self.best_model_name].iloc[0]
        metrics_values = [best_metrics['Accuracy'], best_metrics['Precision'], 
                         best_metrics['Recall'], best_metrics['F1-Score']]
        colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        bars = ax2.bar(metrics, metrics_values, color=colors_bar, alpha=0.8)
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title(f'🏆 Best Model: {self.best_model_name}', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1.1])
        for i, (bar, v) in enumerate(zip(bars, metrics_values)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Confusion Matrix Heatmap
        ax3 = plt.subplot(2, 3, 3)
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], 
                   ax=ax3, cbar_kws={'label': 'Count'}, annot_kws={'size': 14, 'weight': 'bold'})
        ax3.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax3.set_title('🎯 Confusion Matrix', fontsize=14, fontweight='bold')
        
        # 4. Class Distribution
        ax4 = plt.subplot(2, 3, 4)
        class_counts = self.df['label'].value_counts()
        colors_pie = ['#2ecc71', '#e74c3c']
        wedges, texts, autotexts = ax4.pie(class_counts, labels=class_counts.index, 
                                           autopct='%1.1f%%', colors=colors_pie, 
                                           startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
        ax4.set_title('📈 Class Distribution in Dataset', fontsize=14, fontweight='bold')
        
        # 5. Message Length Distribution
        ax5 = plt.subplot(2, 3, 5)
        ham_lengths = self.df[self.df['label']=='ham']['message_length']
        spam_lengths = self.df[self.df['label']=='spam']['message_length']
        ax5.hist(ham_lengths, bins=50, alpha=0.7, label='Ham', color='green', edgecolor='black')
        ax5.hist(spam_lengths, bins=50, alpha=0.7, label='Spam', color='red', edgecolor='black')
        ax5.set_xlabel('Message Length', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax5.set_title('📏 Message Length Distribution', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Feature Importance
        ax6 = plt.subplot(2, 3, 6)
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_names = self.vectorizer.get_feature_names_out()
            indices = np.argsort(importances)[-15:]
            ax6.barh(range(len(indices)), importances[indices], color='teal', alpha=0.8)
            ax6.set_yticks(range(len(indices)))
            ax6.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
            ax6.set_xlabel('Importance', fontsize=12, fontweight='bold')
            ax6.set_title('🔑 Top 15 Feature Importances', fontsize=14, fontweight='bold')
            ax6.grid(axis='x', alpha=0.3)
        elif hasattr(self.best_model, 'coef_'):
            import scipy.sparse
            coef = self.best_model.coef_
            if scipy.sparse.issparse(coef):
                coef = coef.toarray()[0]
            else:
                coef = coef[0]
            feature_names = self.vectorizer.get_feature_names_out()
            indices = np.argsort(np.abs(coef))[-15:]
            ax6.barh(range(len(indices)), coef[indices], color='teal', alpha=0.8)
            ax6.set_yticks(range(len(indices)))
            ax6.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
            ax6.set_xlabel('Coefficient', fontsize=12, fontweight='bold')
            ax6.set_title('🔑 Top 15 Features', fontsize=14, fontweight='bold')
            ax6.grid(axis='x', alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
            ax6.set_title('🔑 Feature Importance', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Visualizations displayed")
    
    def predict(self, messages):
        """Predict spam/ham for new messages"""
        if isinstance(messages, str):
            messages = [messages]
        
        # Preprocess
        cleaned = [self.preprocess_text(msg) for msg in messages]
        
        # Vectorize
        X_vec = self.vectorizer.transform(cleaned)
        
        # Predict
        predictions = self.best_model.predict(X_vec)
        probabilities = self.best_model.predict_proba(X_vec)
        
        results = []
        for msg, pred, prob in zip(messages, predictions, probabilities):
            results.append({
                'message': msg,
                'prediction': 'SPAM' if pred == 1 else 'HAM',
                'confidence': prob[pred] * 100,
                'spam_probability': prob[1] * 100,
                'ham_probability': prob[0] * 100
            })
        
        return results
    
    def create_prediction_widget(self):
        """Create interactive widget for predictions (Colab)"""
        if not IN_COLAB:
            print("Interactive widget only available in Google Colab")
            return
        
        # Create widgets
        text_area = widgets.Textarea(
            value='',
            placeholder='Type your message here...',
            description='Message:',
            layout=widgets.Layout(width='80%', height='100px')
        )
        
        button = widgets.Button(
            description='🔍 Classify Message',
            button_style='primary',
            layout=widgets.Layout(width='200px')
        )
        
        output = widgets.Output()
        
        def on_button_click(b):
            with output:
                clear_output()
                message = text_area.value.strip()
                
                if not message:
                    print("❌ Please enter a message")
                    return
                
                result = self.predict(message)[0]
                
                # Display result with colors
                if result['prediction'] == 'SPAM':
                    color = '#e74c3c'
                    icon = '⚠️'
                else:
                    color = '#2ecc71'
                    icon = '✅'
                
                html_output = f"""
                <div style="padding: 20px; border: 3px solid {color}; border-radius: 10px; background-color: {'#fee' if result['prediction'] == 'SPAM' else '#efe'};">
                    <h2 style="color: {color}; margin: 0;">{icon} {result['prediction']}</h2>
                    <p style="font-size: 18px; margin: 10px 0;"><strong>Confidence:</strong> {result['confidence']:.2f}%</p>
                    <hr style="border-color: {color};">
                    <div style="display: flex; justify-content: space-around; margin-top: 15px;">
                        <div style="text-align: center;">
                            <p style="margin: 5px; font-size: 14px; color: #666;">Spam Probability</p>
                            <p style="margin: 5px; font-size: 24px; font-weight: bold; color: #e74c3c;">{result['spam_probability']:.1f}%</p>
                        </div>
                        <div style="text-align: center;">
                            <p style="margin: 5px; font-size: 14px; color: #666;">Ham Probability</p>
                            <p style="margin: 5px; font-size: 24px; font-weight: bold; color: #2ecc71;">{result['ham_probability']:.1f}%</p>
                        </div>
                    </div>
                </div>
                """
                
                display(HTML(html_output))
        
        button.on_click(on_button_click)
        
        # Display widget
        display(HTML("<h2>🤖 Interactive Spam Classifier</h2>"))
        display(text_area)
        display(button)
        display(output)
        
        # Example buttons
        examples = [
            "Hi, how are you doing today?",
            "URGENT! You've won $1000! Click here now!",
            "Can we meet tomorrow at 3pm?",
            "FREE! Win a brand new iPhone! Text WIN to 12345"
        ]
        
        display(HTML("<h3>Try these examples:</h3>"))
        
        for example in examples:
            example_btn = widgets.Button(
                description=example[:50] + "..." if len(example) > 50 else example,
                layout=widgets.Layout(width='600px', margin='5px')
            )
            
            def make_example_handler(ex):
                def handler(b):
                    text_area.value = ex
                    on_button_click(b)
                return handler
            
            example_btn.on_click(make_example_handler(example))
            display(example_btn)


def main():
    """Main function to run the complete pipeline"""
    # Initialize classifier
    classifier = SpamClassifierColab()
    
    # Upload dataset (in Colab) or use local file
    if IN_COLAB:
        filepath = classifier.upload_dataset()
    else:
        filepath = 'spam.csv'
    
    # Load data
    classifier.load_data(filepath)
    
    # Feature engineering
    classifier.feature_engineering()
    
    # Prepare data
    classifier.prepare_data()
    
    # Train models
    results_df = classifier.train_models()
    
    # Evaluate best model
    cm = classifier.evaluate_best_model()
    
    # Create visualizations
    classifier.plot_results(results_df, cm)
    
    # Demo predictions
    print("\n" + "=" * 70)
    print("🎯 DEMO PREDICTIONS")
    print("=" * 70)
    
    test_messages = [
        "Hi, how are you doing today?",
        "URGENT! You've won $1000! Click here now to claim your prize!",
        "Can we meet tomorrow at 3pm?",
        "FREE! Win a brand new iPhone! Text WIN to 12345",
        "Hey, did you get my email about the meeting?"
    ]
    
    for msg in test_messages:
        result = classifier.predict(msg)[0]
        icon = '⚠️' if result['prediction'] == 'SPAM' else '✅'
        print(f"\n{icon} Message: {msg}")
        print(f"   Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f}%)")
    
    print("\n" + "=" * 70)
    print("✅ PROJECT COMPLETE!")
    print("=" * 70)
    
    # Create interactive widget
    if IN_COLAB:
        print("\n📱 Creating interactive prediction widget...")
        classifier.create_prediction_widget()
    
    return classifier


# Run if executed directly
if __name__ == "__main__":
    classifier = main()
