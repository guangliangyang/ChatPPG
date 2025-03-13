import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 📌 Load the dataset
file_path = "national_illness.csv"  # Change this if the file is elsewhere
df = pd.read_csv(file_path)

# 📌 Drop the 'date' column since it's not numerical
df_numeric = df.drop(columns=['date'])

# 📌 Apply MinMaxScaler
minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df_numeric), columns=df_numeric.columns)

# 📌 Apply StandardScaler
standard_scaler = StandardScaler()
df_standard = pd.DataFrame(standard_scaler.fit_transform(df_numeric), columns=df_numeric.columns)

# 📌 Display the transformed data
print("🔹 Original Data (First 5 Rows):\n", df_numeric.head())
print("\n🔹 MinMax Scaled Data (First 5 Rows):\n", df_minmax.head())
print("\n🔹 Standard Scaled Data (First 5 Rows):\n", df_standard.head())

# 📌 Plot comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# 📊 1. Original Data Distribution
sns.boxplot(data=df_numeric, ax=axes[0])
axes[0].set_title("📌 Original Data Distribution")

# 📊 2. MinMaxScaler Data Distribution
sns.boxplot(data=df_minmax, ax=axes[1])
axes[1].set_title("📌 MinMax Scaled Data Distribution")

# 📊 3. StandardScaler Data Distribution
sns.boxplot(data=df_standard, ax=axes[2])
axes[2].set_title("📌 Standard Scaled Data Distribution")

plt.tight_layout()
plt.show()



# MinMaxScaler is useful when the data has a bounded range or when the distribution is not Gaussian. For example, in image processing, pixel values are typically in the range of 0-255. Scaling these values using MinMaxScaler ensures that the values are within a fixed range and contributes equally to the analysis. Similarly, when dealing with non-Gaussian distributions such as a power-law distribution, MinMaxScaler can be used to ensure that the range of values is scaled between 0 and 1.

# StandardScaler is useful when the data has a Gaussian distribution or when the algorithm requires standardized features. For example, in linear regression, the features need to be standardized to ensure that they contribute equally to the analysis. Similarly, when working with clustering algorithms such as KMeans, StandardScaler can be used to ensure that the features are standardized and contribute equally to the analysis.


#
# I generally decide from the two algorithms as follows:
#
# If the range of data value is known (like Marks- general range (0-100))then Min-Max Scaler, else Standard Scaler
# If the number of data features is high, then Standard Scalar, else Min-Max
# If the number of data features are in varied ranges or different units, then Standard Scalar, else Min-Max
# If the feature has outliers that can't be treated, then Min-Max Scaler, else Standard Scaler. Min-Max Scaler doesn't reduce the importance of outliers


# StandardScaler: Assumes that data has normally distributed features and will scale them to zero mean and 1 standard deviation. Use StandardScaler() if you know the data distribution is normal. For most cases StandardScaler would do no harm. Especially when dealing with variance (PCA, clustering, logistic regression, SVMs, perceptron's, neural networks) in fact Standard Scaler would be very important. On the other hand it will not make much of a difference if you are using tree based classifiers or regressors.
#
# MinMaxScaler : This will transform each value in the column proportionally within the range [0,1]. This is quite acceptable in cases where we are not concerned about the standardization along the variance axes. e.g. image processing or neural networks expecting values between 0 to 1.
#
