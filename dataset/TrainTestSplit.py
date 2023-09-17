import pandas as pd

# splitting the data into train and validate sets in the ratio 80:20

data = pd.read_csv('dataset/sentences.csv')

data = data.sample(frac=1, random_state=42)

train_ratio = 0.8
train_size = int(len(data) * train_ratio)

train_data = data.iloc[:train_size]
validate_data = data.iloc[train_size:]

train_data.to_csv('dataset/train_dataset.csv', index=False)
validate_data.to_csv('dataset/validate_dataset.csv', index=False)