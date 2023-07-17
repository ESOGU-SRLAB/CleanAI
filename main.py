import os
import os.path
import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms

from analyzer import Analyzer
from image_loader import ImageLoader


class NeuralNetwork(nn.Module):
    # Burda modeli tanımlıyoruz. Eğer model yerel makine üzerinden yüklenecekse
    # model tanımına ihtiyaç bulunmaktadır.
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def analyze_neural_network():
    # Bu fonksiyon herhangi bir modelin nasıl analiz edileceğini göstermektedir.
    # Burada bulunan işlemler direk 'main()' fonksiyonu içerisinde de tanımlanabilirdi.
    # Fakat kodun okunabilirliği açısından bu şekilde bir yapı tercih edilmiştir.
    # Fonksiyon içerisinde modelin yüklenmesi, veri setinin uygun hale getirilebilmesi için
    # transform değişkeninin tanımlanması ve analiz işlemleri esnasında kullanılacak parametre
    # değerlerinin tanımlanması işlemleri yapılmaktadır.
    model = NeuralNetwork()
    model = torch.load("./model_fashion_1.pth")  # Modelin yüklenmesi

    # 'transform' değişkeninin tanımlanması modelden modele ve veri setinden veri setine farklılık
    # gösterebilmektedir. Bu yüzden her model için ayrı ayrı tanımlanması gerekmektedir.
    # Burada bulunan transform değişkeni MNIST veri seti için tanımlanmıştır.
    # Veri seti ve model için uygun olan 'transform' değişkeninin tanımlanması için
    # modele ait dökümantasyonun incelenmesi ve tavsiye edilen değişkenin kullanılması
    # gerekmektedir.
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )  # Veri setinin uygun hale getirilmesi için transform değişkeninin tanımlanması
    image_loader = ImageLoader("./test", transform)  # Veri setinin yüklenmesi

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [
        (0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
    ]  # Analiz işlemleri esnasında kullanılacak parametre değerlerinin tanımlanması

    analyze = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
    )  # Analyzer sınıfının çağırılması
    analyze.analyze()


def analyze_maxvit():
    model = models.maxvit_t(pretrained=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Görüntü boyutunu yeniden şekillendir
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_loader = ImageLoader("./maxvit_dataset", transform)

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    analyzer = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
        False,
    )
    analyzer.analyze()


def analyze_resnet18():
    model = models.resnet18(pretrained=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_loader = ImageLoader("./maxvit_dataset", transform)

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    analyzer = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
        False,
    )
    analyzer.analyze()


def analyze_resnet34():
    model = models.resnet34(pretrained=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_loader = ImageLoader("./maxvit_dataset", transform)

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    analyzer = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
        False,
    )
    analyzer.analyze()


def analyze_resnet50():
    model = models.resnet50(pretrained=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_loader = ImageLoader("./maxvit_dataset", transform)

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    analyzer = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
        False,
    )
    analyzer.analyze()


def analyze_resnet101():
    model = models.resnet101(pretrained=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_loader = ImageLoader("./maxvit_dataset", transform)

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    analyzer = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
        False,
    )
    analyzer.analyze()


def analyze_resnet152():
    model = models.resnet152(pretrained=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_loader = ImageLoader("./maxvit_dataset", transform)

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    analyzer = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
        False,
    )
    analyzer.analyze()


def analyze_alexnet():
    model = models.alexnet(pretrained=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_loader = ImageLoader("./maxvit_dataset", transform)

    how_many_samples = 50
    th_cov_val = 0.75
    value_cov_th = 0.75
    top_k_val = 3
    node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

    analyzer = Analyzer(
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
        False,
    )
    analyzer.analyze()


def main():
    # analyze_neural_network()
    # analyze_maxvit()
    # analyze_alexnet()
    # analyze_resnet18()
    # analyze_resnet34()
    # analyze_resnet50()
    # analyze_resnet101()
    analyze_resnet152()


if __name__ == "__main__":
    main()
