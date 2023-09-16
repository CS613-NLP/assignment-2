import pandas as pd

data = pd.read_csv('sentences.csv')

data = data.sample(frac=1, random_state=42)
print('Number of sentences:', len(data))
train_ratio = 0.8
train_size = int(len(data) * train_ratio)

print('Train size:', train_size)
print('Validate size:', len(data) - train_size)

train_data = data.iloc[:train_size]
validate_data = data.iloc[train_size:]

train_data.to_csv('train_dataset.csv', index=False)
validate_data.to_csv('validate_dataset.csv', index=False)