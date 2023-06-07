import random
import matplotlib.pyplot as plt
from custom_dataset import CustomDataset

# FashionMNIST veri setinin CSV dosyasının yolu
data_file = "fashion-mnist_test.csv"

# CustomDataset sınıfını kullanarak veri setini oluşturun
dataset = CustomDataset(data_file)

# Rastgele bir indeks seçin
random_index = random.randint(0, len(dataset) - 1)

# Seçilen örneği alın
sample, label = dataset.get_random_data()

# Birden fazla örnek al
samples = dataset.get_random_datas(3)

# Resmi ekrana bastır
plt.imshow(sample.view(28, 28), cmap='gray')
plt.title('Label: {}'.format(label))
plt.axis('off')
plt.show()

# Resimleri ekrana bastır
for sample in samples:
    plt.imshow(sample[0].view(28, 28), cmap='gray')
    plt.title('Label: {}'.format(sample[1]))
    plt.axis('off')
    plt.show()
