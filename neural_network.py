import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler


def load_data():
    print("Loading data from CSV files...")
    csv_files = glob.glob("harth/*.csv")
    dataframes = {}
    for file in csv_files:
        df = pd.read_csv(file)
        key = file.split("\\")[-1].split(".")[0]
        dataframes[key] = df

    whole_data = pd.concat(dataframes.values())
    whole_data.drop(["timestamp", "index", "Unnamed: 0"], axis=1, inplace=True)
    print("Data loaded and combined successfully.")
    print(whole_data.head())
    return whole_data


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=20):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


def main():
    whole_data = load_data()

    # Assuming the last column is the label
    X = whole_data.iloc[:, :-1].values
    y = whole_data.iloc[:, -1].values

    # Create a mapping from original labels to new labels starting from 0
    unique_labels = sorted(set(y))
    label_mapping = {
        original_label: idx for idx, original_label in enumerate(unique_labels)
    }
    print("Label mapping:", label_mapping)

    # Apply the mapping to labels
    y_mapped = [label_mapping[label] for label in y]

    # Check number of unique classes
    num_classes = len(label_mapping)  # Ensure this matches the output layer size

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y_mapped, dtype=torch.long)  # Use mapped labels

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = X.shape[1]

    # Correct number of classes passed here
    model = SimpleNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_model(model, train_loader, criterion, optimizer, device)

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
