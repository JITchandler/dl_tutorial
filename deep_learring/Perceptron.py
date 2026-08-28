import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

batch_size = 256

# 修复变量名冲突
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_iter = DataLoader(datasets.FashionMNIST("./data", True, transform, download=True),
                        batch_size, shuffle=True)
test_iter = DataLoader(datasets.FashionMNIST("./data", False, transform, download=True),
                       batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 关键：去掉*0.1，使用标准初始化
w1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.5)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

w2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.5)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [w1, b1, w2, b2]


def relu(x):
    return torch.maximum(torch.zeros_like(x), x)


def net(x):
    x = x.reshape(-1, num_inputs)
    h = relu(x @ w1 + b1)
    return h @ w2 + b2


loss = nn.CrossEntropyLoss(reduction='none')

# 关键：提高学习率到0.5-1.0
num_epochs, lr = 20, 0.01  # 从0.01提高到0.5
updater = torch.optim.SGD(params, lr)


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(num_epochs):
        train_loss, train_acc, total = 0.0, 0.0, 0

        for x, y in train_iter:
            y_hat = net(x)
            l = loss(y_hat, y)

            updater.zero_grad()
            l.sum().backward()

            # 可选：打印梯度范数检查是否更新
            # grad_norm = sum(p.grad.norm().item() for p in params if p.grad is not None)
            # print(f"Gradient norm: {grad_norm:.6f}")

            updater.step()

            train_loss += l.sum().item()
            train_acc += (y_hat.argmax(dim=1) == y).sum().item()
            total += x.shape[0]

        train_loss /= total
        train_acc /= total

        test_acc, test_total = 0, 0
        with torch.no_grad():
            for x, y in test_iter:
                y_hat = net(x)
                test_acc += (y_hat.argmax(dim=1) == y).sum().item()
                test_total += x.shape[0]
            test_acc /= test_total

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"第 {epoch + 1} 轮 | "
              f"训练损失: {train_loss:.3f} | "
              f"训练准确率: {train_acc:.3f} | "
              f"测试准确率: {test_acc:.3f}")

    # 绘图
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, color="#e74c3c", marker="o")
    plt.xlabel("训练轮数")
    plt.ylabel("损失值")
    plt.title("训练损失下降曲线")
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accs, color="#27ae60", marker="o", label="训练准确率")
    plt.plot(range(1, num_epochs + 1), test_accs, color="#3498db", marker="s", label="测试准确率")
    plt.xlabel("训练轮数")
    plt.ylabel("准确率")
    plt.title("训练/测试准确率上升曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


train(net, train_iter, test_iter, loss, num_epochs, updater)