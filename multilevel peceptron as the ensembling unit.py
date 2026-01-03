import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Define the Ensemble Model

class EnsembleNet(nn.Module):
    def __init__(self):
        super(EnsembleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# 2. Generate Dummy Training Data

def generate_dummy_data(n=1000):
    X = []
    y = []
    for _ in range(n):
        p_cnn = np.random.rand()
        p_vit = np.random.rand()
        g = np.random.rand()
        p_final = g * p_cnn + (1 - g) * p_vit
        label = 1 if p_final + np.random.normal(0, 0.05) > 0.5 else 0
        X.append([p_cnn, p_vit, g])
        y.append([label])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# 3. Train the Ensemble Model

def train_ensemble_model(model, X, y, epochs=100, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X).squeeze()
        loss = criterion(outputs, y.squeeze())
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            preds = (outputs > 0.5).float()
            acc = (preds == y.squeeze()).float().mean()
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {acc.item():.4f}")


# 4. Inference Function
def predict_with_ensemble(model, p_cnn, p_vit, g):
    model.eval()
    input_tensor = torch.tensor([[p_cnn, p_vit, g]], dtype=torch.float32)
    with torch.no_grad():
        p_final = model(input_tensor).item()
    preferred_model = "CNN" if g >= 0.5 else "ViT"
    prediction = "Deepfake" if p_final > 0.5 else "Real"

    return {
        "P_CNN": round(p_cnn, 4),
        "P_ViT": round(p_vit, 4),
        "Gating_Score": round(g, 4),
        "P_Final": round(p_final, 4),
        "Preferred_Model": preferred_model,
        "Prediction": prediction
    }


# 5. Main Execution
if __name__ == "__main__":
    # Step 1: Generate training data
    X_train, y_train = generate_dummy_data(n=1000)

    # Step 2: Initialize and train model
    model = EnsembleNet()
    train_ensemble_model(model, X_train, y_train, epochs=100)

    # Step 3: Inference on a sample input
    sample_input = {
        "p_cnn": 0.72,
        "p_vit": 0.61,
        "g": 0.68
    }

    result = predict_with_ensemble(model, **sample_input)
    print("\n Inference Result:")
    for k, v in result.items():
        print(f"{k}: {v}")
Epoch 0: Loss = 0.6931, Accuracy = 0.5050
...
Epoch 90: Loss = 0.3124, Accuracy = 0.8740
Epoch 99: Loss = 0.2981, Accuracy = 0.8820


#Inference Result:
#P_CNN: 0.72
#P_ViT: 0.61
#Gating_Score: 0.68
#P_Final: 0.6932
#Preferred_Model: CNN
#Prediction: Deepfake


