import torch
import torch.nn as nn
from torch.nn import Sequential
from matplotlib import pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

path = "./data/CELLS/"
classes = ["linba", "zhongxingli"]

batch_size = 64
epochs = 50
lr = 0.001

test_path = "./data/Pictures/"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.CenterCrop(128),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

data_train = datasets.ImageFolder(root=path, transform=transform)
train_loader = DataLoader(data_train, batch_size=64, shuffle=True)

data_test = datasets.ImageFolder(root=test_path, transform=transform)
test_loader = DataLoader(data_test, batch_size=64, shuffle=True)

# -----------------------展示数据集---------------------------
images, labels = next(iter(train_loader))
img = images[0].numpy().transpose(1, 2, 0)
plt.imshow(img)
plt.title(labels[0])
plt.show()


# -----------------------展示数据集---------------------------


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv4 = Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.dense = Sequential(
            nn.Linear(7 * 7 * 256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = x4.view(-1, 7 * 7 * 256)
        out = self.dense(x5)
        return out


def get_variable(x):
    x = torch.autograd.Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


cnn = CNN()
if torch.cuda.is_available():
    cnn = cnn.cuda()

lossF = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

cnn.train()

loss_pth = 999999999.99
i_pth = 0

for epoch in range(epochs):
    running_loss = 0.0
    running_correct = 0.0
    print("Epochs [{}/{}]".format(epoch, epochs))

    for data in train_loader:
        X_train, y_train = data
        X_train, y_train = get_variable(X_train), get_variable(y_train)
        outputs = cnn(X_train)
        _, predict = torch.max(outputs.data, 1)
        # ----------------------------------
        optimizer.zero_grad()
        loss = lossF(outputs, y_train)
        loss.backward()
        optimizer.step()
        # ----------------------------------
        running_loss += loss.item()
        running_correct += torch.sum(predict == y_train.data)

    testing_correct = 0.0

    for data in test_loader:
        X_test, y_test = data
        X_test, y_test = get_variable(X_test), get_variable(y_test)
        outputs = cnn(X_test)
        _, predict = torch.max(outputs.data, 1)
        testing_correct += torch.sum(predict == y_test.data)

    print("Loss: {}    Training Accuracy: {}%    Testing Accuracy:{}%".format(
        running_loss,
        100 * running_correct / len(data_train),
        100 * testing_correct / len(data_test)
    ))

    if running_loss < loss_pth:
        loss_pth = running_loss
        torch.save(cnn, "./models/cell_classify_%d.pth" % i_pth)
        i_pth = i_pth + 1

torch.save(cnn, "cell_classify.pth")
