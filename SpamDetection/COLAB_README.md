SMS Spam Classifier 
A complete machine learning project optimized for Google Colab that classifies SMS messages as spam or ham with 98%+ accuracy.



Files Included
Spam_Classifier_Colab.ipynb -
spam_classifier_colab.py 
COLAB_README.md 

St

Install Packages (automatic)



Upload Dataset

Click the upload button when prompted
Select your spam.csv file

Train Models

Run all cells
Watch as 5 models train and compete
See which one performs best

Test Messages

Use the interactive widget
Type any message
Get instant predictions with confidence scores
What You'll Get
Model Performance
Typical results:

Best Model: SVM or Random Forest
Accuracy: 97-98%
Precision: 98-100%
Recall: 85-90%
F1-Score: 91-95%
Visualizations
Model comparison chart
 Best model metrics
 Confusion matrix
Class distribution
Message length analysis
 Top predictive features
Interactive Widget
Beautiful UI with color-coded results
Real-time predictions
Confidence scores
Spam/Ham probabilities
Example messages to try
Example Usage
In the Interactive Widget
Ham (Legitimate):

Message: "Hi, how are you doing today?"

Result: ✅ HAM (Confidence: 99.82%)

Spam:

Message: "URGENT! You've won $1000! Click here now!"

Result: ⚠️ SPAM (Confidence: 100.00%)
Programmatically
# Predict a message

result = predict_message("Can we meet tomorrow?")

# Result:

# {

#     'prediction': 'HAM',

#     'confidence': 99.89,

#     'spam_prob': 0.11,

#     'ham_prob': 99.89

# }
Interactive prediction widget
Customization
Change Models
Add or remove models in the training section:

models = {

    'Naive Bayes': MultinomialNB(),

    'Your Model': YourClassifier()

}
Adjust Vectorization
Modify TF-IDF parameters:

vectorizer = TfidfVectorizer(

    max_features=5000,  # More features

    ngram_range=(1, 3)  # Include trigrams

)
Change Test Size
Modify train/test split:

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3  # 30% test data

)

 Performance Tips
For Better Accuracy:
Use more training data
Experiment with n-gram ranges
Try different max_features
Tune model hyperparameters
For Faster Training:
Reduce max_features to 1000-2000
Use simpler models (Naive Bayes)
Reduce n-gram range to (1, 1)


Additional Features
Save Trained Model
import joblib

# Save model

joblib.dump({

    'model': best_model,

    'vectorizer': vectorizer

}, 'spam_model.pkl')

# Download from Colab

from google.colab import files

files.download('spam_model.pkl')
Batch Predictions
# Predict multiple messages

messages = [

    "Message 1",

    "Message 2",

    "Message 3"

]

for msg in messages:

    result = predict_message(msg)

    print(f"{msg}: {result['prediction']}")
Use Cases
SMS spam filtering
Email spam detection
Social media message filtering
Chatbot content moderation
Educational ML projects
Data science portfolios

