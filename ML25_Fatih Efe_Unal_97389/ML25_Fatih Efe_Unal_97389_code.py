import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Preprocess Blood Pressure column (convert '120/80' to average value)
def parse_bp(bp_str):
    try:
        systolic, diastolic = map(int, bp_str.split('/'))
        return (systolic + diastolic) / 2
    except:
        return np.nan

df['Blood Pressure'] = df['Blood Pressure'].apply(parse_bp)

# Select relevant features and drop missing values
features = ['BMI Category', 'Stress Level', 'Occupation', 'Blood Pressure']
target = 'Sleep Disorder'
df = df[features + [target]].dropna()

# Encode categorical variables
label_encoders = {}
for col in ['Occupation', 'BMI Category', 'Sleep Disorder']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Convert to numpy arrays
X = df[features].astype(float).values
y = df['Sleep Disorder'].values

# Standardize features ( centering data around 0 with unit variance )
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  #data split to 80/20
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32) #neural networks do math with float
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Define neural network class
class SleepNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SleepNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32) #First hidden layer with 32 neurons
        self.dropout = nn.Dropout(0.2) #randomly turning off 20% of neurons during training to prevent overfitting
        self.fc2 = nn.Linear(32, 16) #Second hidden layer
        self.fc3 = nn.Linear(16, output_dim) #Final layer that outputs raw scores for each class

    def forward(self, x):   #applying activation function (relu) after each hidden layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x) #for regularization
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
input_dim = X.shape[1]
output_dim = len(np.unique(y))
model = SleepNet(input_dim, output_dim)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
epochs = 100
train_acc = []
val_acc = []
train_loss = []
val_loss = []
for epoch in range(epochs):
    model.train()
    correct = total = 0
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
        epoch_loss += loss.item()
    train_accuracy = correct / total
    train_acc.append(train_accuracy)
    train_loss.append(epoch_loss / len(train_loader))

    # Validation
    model.eval()
    correct = total = 0
    epoch_val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            epoch_val_loss += loss.item()
    val_accuracy = correct / total
    val_acc.append(val_accuracy)
    val_loss.append(epoch_val_loss / len(test_loader))

    print(f"Epoch {epoch+1}/{epochs}, Train Acc: {train_accuracy:.2f}, Val Acc: {val_accuracy:.2f}")

# Plot training history
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()

# Permutation feature importance
model.eval()
base_accuracy = val_acc[-1]
importances = []
for i in range(X.shape[1]):
    X_test_shuffled = X_test.copy()
    np.random.shuffle(X_test_shuffled[:, i])
    X_tensor = torch.tensor(X_test_shuffled, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
        acc = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    drop = base_accuracy - acc
    importances.append(drop)

# Plot feature importances
plt.figure(figsize=(8, 5))
plt.bar(features, importances)
plt.title("Permutation Feature Importance")
plt.ylabel("Accuracy Drop")
plt.xlabel("Feature")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Feature histograms
df.hist(figsize=(10, 6))
plt.tight_layout()
plt.suptitle("Feature Distributions", y=1.02)
plt.show()

# Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()