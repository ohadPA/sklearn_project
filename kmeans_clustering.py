import utils
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

# Load the data into a pandas DataFrame
df = pd.read_csv("ffp_train.csv")

# Prepare the data for the KMeans model
features = df.drop("ID", axis=1).drop("CUSTOMER_GRADE", axis=1).drop("BUYER_FLAG", axis=1) # Features without the binary score column

# Use SelectKBest to select the best features based on chi-squared test
selector = SelectKBest(chi2, k=5)
selected_features = selector.fit_transform(features, df["BUYER_FLAG"])

# Train the KMeans model with the features
kmeans = KMeans(n_clusters=2, n_init=5)
kmeans.fit(selected_features)
print("selected_features => ", selector.get_feature_names_out())

# Predict the cluster each customer belongs to
predictions = kmeans.predict(selected_features)

# Assign the cluster labels to the customers
df["cluster"] = predictions

# Count the number of customers in each cluster
clusterCounts = df["cluster"].value_counts()
print("cluster counts => ", clusterCounts)

buyerCounts = df["BUYER_FLAG"].value_counts()
print("buyer counts => ", buyerCounts)

# Get the mean score of each cluster
cluster_means = df.groupby("cluster").mean()["BUYER_FLAG"]
print(cluster_means)


# Print the accuracy of the model
print("Accuracy: " + str(accuracy_score(df["BUYER_FLAG"], df["cluster"])*100))

tn, fp, fn, tp = confusion_matrix(df["BUYER_FLAG"], df["cluster"], labels=[0, 1]).ravel()
print("True Positive: " + str(tp))
print("False Positive: " + str(fp))
print("False Negative: " + str(fn))
print("True Negative: " + str(tn))

print("MSE " + str(np.sqrt(mean_squared_error(df["BUYER_FLAG"], df["cluster"]))))


# Load the data into a pandas DataFrame
df_test = pd.read_csv("ffp_rollout_X.csv")

cols_idxs = selector.get_support(indices=True)
selected_subset_of_test_file = df_test.iloc[:,cols_idxs]

# Predict the cluster each customer belongs to
predictions = kmeans.predict(selected_subset_of_test_file)

print("selected_subset_of_test_file -> ", selected_subset_of_test_file)
# Assign the cluster labels to the customers
df_test["BUYER_FLAG_2"] = predictions

# Export to csv
output_df = utils.create_output_df(predictions)
utils.create_output_csv("classifier_output.csv", output_df)