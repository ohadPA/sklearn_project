import pandas
import numpy as np
import utils 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split

#---Training Phase----#
# Load the data into a Pandas DataFrame
df = pandas.read_csv('text_training.csv')

# Create the bag of words using the CountVectorizer
stop_words=['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
vectorizer = CountVectorizer(stop_words=stop_words, max_features=1000)

# Create the target variable
target_var = df['rating']

# Split the data into train and test sets
bag_of_words_train, bag_of_words_test, target_var_train, target_var_test = train_test_split(df, target_var, test_size=0.3)

# Train the model using a Logistic Regression classifier
clf = LogisticRegression(max_iter=2000)
clf.fit(bag_of_words_train, target_var_train)

# Make predictions on the test set
target_var_pred = clf.predict(bag_of_words_test)

# Print the accuracy of the model
print("Accuracy: " + str(accuracy_score(target_var_test, target_var_pred)*100))

tn, fp, fn, tp = confusion_matrix(target_var_test, target_var_pred, labels=[0, 1]).ravel()
print("True Positive: " + str(tp))
print("False Positive: " + str(fp))
print("False Negative: " + str(fn))
print("True Negative: " + str(tn))

print("MSE " + str(np.sqrt(mean_squared_error(target_var_test, target_var_pred))))
#---Prediction Phase----#

# Load the rollout data into a Pandas DataFrame
rollout_df = pandas.read_csv('text_rollout_X.csv')

# Make predictions on the rollout
target_var_rollout = clf.predict(rollout_df)

# Export to csv
output_df = utils.create_output_df(target_var_rollout)
utils.create_output_csv("vectorizer_output.csv", output_df)
