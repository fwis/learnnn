import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms

# 定义FFFullConnect层
class FFFullConnect(nn.Module):
    def __init__(self, input_dim, output_dim, threshold=1.5, learning_rate=0.03):
        super(FFFullConnect, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_metric = nn.MSELoss()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x_norm = torch.norm(x, p='fro', dim=1, keepdim=True) + 1e-4
        x_dir = x / x_norm
        res = self.linear(x_dir)
        return self.relu(res)

    def forward_forward(self, x_pos:torch.Tensor, x_neg:torch.Tensor) -> list[torch.Tensor]:
        self.optimizer.zero_grad()
        g_pos = torch.mean(self.forward(x_pos) ** 2, dim=1)
        g_neg = torch.mean(self.forward(x_neg) ** 2, dim=1)
        loss_pos = torch.log(1 + torch.exp(-g_pos + self.threshold))
        loss_neg = torch.log(1 + torch.exp(g_neg - self.threshold))
        
        loss = torch.cat([loss_pos, loss_neg]).mean()
        
        loss.backward()
        self.optimizer.step()
        
        return self.forward(x_pos), self.forward(x_neg), loss

# 定义FFNetwork自定义模型
class FFNetwork(nn.Module):
    def __init__(self, dims):
        super(FFNetwork, self).__init__()
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(FFFullConnect(dims[i], dims[i+1]).to(device))

    def overlay_y_on_x(self, x:torch.Tensor, y:torch.Tensor, classes=10)->torch.Tensor:
        x[:classes] = 0
        x[y] = x.max()
        return x

    def predict_one_sample(self, x:torch.Tensor):
        x = x.view(-1)
        goodness_per_label = []
        for label in range(10):
            x_label = x.clone()
            x_label[:10] = 0
            x_label = self.overlay_y_on_x(x_label, label)
            h = x_label.unsqueeze(0)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness.append(torch.mean(h ** 2, dim=1))
            goodness_per_label.append(sum(goodness))
        goodness_per_label = torch.cat(goodness_per_label, dim=0)
        return torch.argmax(goodness_per_label).item()

    def predict(self, data):
        self.eval()
        preds = []
        with torch.no_grad():
            for x in data:
                preds.append(self.predict_one_sample(x))
        return np.array(preds)

    def train_step(self, x:torch.Tensor, y:torch.Tensor):
        x = x.view(x.size(0), -1).to(device)
        
        x_pos = x.clone()
        for i in range(x.size(0)):
            x_pos[i] = self.overlay_y_on_x(x_pos[i], y[i])
        
        y_shuffled = y[torch.randperm(y.size(0))]
        x_neg = x.clone()
        for i in range(x.size(0)):
            x_neg[i] = self.overlay_y_on_x(x_neg[i], y_shuffled[i])
        
        for layer in self.layers:
            x_pos, x_neg, loss = layer.forward_forward(x_pos, x_neg)
            
        return loss.item()


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

batch_size = 128

# 训练模型
model = FFNetwork(dims=[784, 2000, 2000]).to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 8
for epoch in range(num_epochs):
    loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        loss = model.train_step(images, labels)
        
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}")
    loss = 0

# 测试模型
test_images, test_labels = next(iter(test_loader))
test_images, test_labels = test_images.to(device), test_labels.to(device)
test_preds = model.predict(test_images.view(test_images.size(0), -1).to(device))

accuracy = accuracy_score(test_preds, test_labels.cpu().numpy())
print(f"Test Accuracy: {accuracy * 100:.2f}%")