import pandas as pd

# Load dataset
data = pd.read_csv("StudentsPerformance.csv")

# Create target column 'pass'
data['average_score'] = (data['math score'] + data['reading score'] + data['writing score']) / 3
data['pass'] = (data['average_score'] >= 50).astype(int)

# Drop 'average_score' column (not needed as input)
data = data.drop(columns=['average_score'])

# One-hot encode categorical features
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
data = pd.get_dummies(data, columns=categorical_cols)

# Separate features and target
X = data.drop(columns=['pass'])
y = data['pass']

print("Features shape:", X.shape)
print("Target distribution:\n", y.value_counts())
