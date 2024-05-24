import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

batch_size = 16 # ���������� ����������� ������� ����� �������� �� �����

# ���������� �������������� ��� ������������ ������
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# ��������� �������� ����� ������ MNIST
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# ������� DataLoader ��� ��������� ������ ������
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# �������� ���� ������ ����������� � �����
data_iter = iter(test_loader)
images, labels = next(data_iter)

# ������� ��� ����������� �����������
def imshow(img):
    img = img / 2 + 0.5  # ������������� �����������
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# ���������� �����������
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % labels[j].item() for j in range(batch_size)))