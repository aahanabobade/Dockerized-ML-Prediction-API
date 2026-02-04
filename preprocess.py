import pandas as pd

data = pd.read_csv("StudentsPerformance.csv")

# Create target
data['average_score'] = (
    data['math score'] + data['reading score'] + data['writing score']
) / 3

data['pass'] = (data['average_score'] >= 50).astype(int)

# Drop helper column
data.drop(columns=['average_score'], inplace=True)

# One-hot encode categorical columns
categorical_cols = [
    'gender',
    'race/ethnicity',
    'parental level of education',
    'lunch',
    'test preparation course'
]

data = pd.get_dummies(data, columns=categorical_cols)

# 🔥 THIS WAS MISSING
data.to_csv("student_data_preprocessed.csv", index=False)

print("Preprocessing complete. File saved as student_data_preprocessed.csv")
