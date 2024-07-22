# CleanAI - Evaluation of Deep Neural Network Quality by CleanAI Coverage Metrics Library

![lang](https://img.shields.io/github/languages/top/inomuh/imfit)

---

## Contents

The document includes the following titles:

- [What is CleanAI](#what-is-cleanai)
- [Coverage Metrics](#coverage-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Sample Analysis Reports (Case Study and Output Results on Different ResNet Models)](#sample-analysis-reports-case-study-and-output-results-on-different-resnet-models)
- [Collobrators](#collobrators)
- [Credits](#credits)
- [License](#license)

---

## What is CleanAI?

CleanAI is a white-box testing library that utilizes coverage metrics to evaluate the structural analysis, quality, and reliability parameters of DNN models. The library incorporates eleven coverage test methods and enables developers to perform essential model analyses and generate output reports. By using CleanAI, developers can effectively evaluate the quality of DNNs and make informed decisions.

CleanAI is dedicated to enhancing the quality assurance of AI systems through effective testing methodologies and valuable insights for developers. Our research aims to advance AI safety, promote responsible development and deployment, and mitigate risks associated with AI technologies. By improving reliability and quality assurance, we can foster trust and ensure the ethical and effective use of AI across domains, ultimately enhancing its societal impact. The utilization of our CleanAI coverage metrics library aims to enhance reliability while addressing ethical concerns. Through comprehensive testing and analysis, we aim to minimize risks and improve decision-making, reliability, explainability, and accountability in AI systems, thus enhancing AI safety.

Within the scope of this study, our CleanAI library measures various coverage metrics to determine the quality of deep neural networks. These coverage metrics include **Neuron Coverage**, **Threshold Coverage**, **Sign Coverage**, **Value Coverage**, **Sign-Sign Coverage**, **Value-Sign Coverage**, **Sign-Value Coverage**, **Value-Value Coverage**, **Neuron Boundary Coverage**, **Multisection Neuron Coverage**, and **Top K Neuron Coverage**.

---

## Coverage Metrics

The CleanAI tool deals with 3 different types of metrics: **Activation Metrics**, **Boundary Metrics**, and **Interaction Metrics**. Activation metrics focus on evaluating the activation status and values of neurons within the neural network. Boundary metrics evaluate the activation boundaries of neurons, helping to understand the conditions under which neurons become active or inactive. Interaction Metrics focus on the relationships and interactions between pairs of neurons.

### Activation Metrics

**Neuron Coverage:** This coverage criterion tracks the activation status of each neuron in a neural network. It aims to determine which neurons are activated and which ones are not. In our implementation of the Neuron Coverage criterion, we have made a difference compared to the literature. The distinction lies in how we calculate the activation status of neurons. Instead of directly comparing the after values of each neuron with a fixed threshold, we take a different approach. After each activation function, we compute the average of the resulting "after value" for each layer. Then, we check if the after value of each neuron is greater than the calculated average for its corresponding layer. Neurons with after values greater than the average are considered active. By calculating the average of all neurons in each layer and comparing individual neuron values to their respective average, we determine the activation status of each neuron. This approach allows us to assess the proportion of activated neurons in the neural network based on the after values relative to the layer averages.

**Threshold Coverage:** The threshold coverage value of a model is the ratio of the number of covered neurons (number of neurons greater than the threshold value) to the total number of neurons in the model.

**Top-K Neuron Coverage:** Top-K Neuron Coverage (TKNC) is a metric used to evaluate the activation patterns and coverage of neurons in a deep neural network (DNN). It measures the percentage of neurons that are activated for a given set of input samples. The idea behind TKNC is to assess how well a set of input samples can activate different neurons in the network. **_How is it calculated?_** TKNC travels through all layers on a model one by one and ranks the neuron values of each layer in order from largest to smallest. Then it takes k neurons in each layer and adds it to a list. It then creates a value called 'TKNC Sum', which represents the sum of neurons in this list. The 'Number of Selected Neurons' value shows how many neurons were selected on the whole model as a result of k neurons from each layer. The 'Mean of Top-K Neurons' value shows the ratio of the 'TKNC Sum' value to the 'Number of Selected Neurons' value.

**Sign Coverage:** When given two different test inputs, it checks whether the signs of a specific neuron's value after the activation function are the same. If the signs are not the same, the counter is incremented.

**Value Coverage:** When given two different test inputs, it checks whether the difference between the values of a specific neuron after the activation function is greater than the given threshold value. If the difference is greater than the threshold value, the counter is incremented.

### Boundary Metrics

**Neuron Boundary Coverage:** Neuron Boundary Coverage (NBC) is a metric used to evaluate the coverage of decision boundaries in a deep neural network (DNN). It measures the percentage of decision boundaries in the network that have been activated or crossed by the input samples. **_How is it calculated?_** NBC receives a random set of inputs from the user, and as a result of these inputs, it determines the maximum and minimum interval value for each layer. Then, for the input data to be checked, it is checked whether each neuron belonging to each layer is within the maximum and minimum range of this layer. If it is within this range, the 'NBC Counter' value is increased by one.

**Multisection Neuron Coverage:** Multisection Neuron Coverage (MNC) specifically focuses on assessing the coverage of individual neurons within the model. The goal of MNC is to evaluate the degree to which the decisions made by individual neurons have been exercised by the test cases. It helps identify potential shortcomings in the model's behavior and reveal areas that may require further testing. It provides the user with the information of how many neurons are found according to the threshold value ranges given by the user. **_How is it calculated?_** The MNC receives threshold ranges from the user. Then, it evaluates all the neurons on the model and checks whether each neuron is within these threshold ranges. If the corresponding neuron is within this threshold value, it increases the 'MNC Counter' value found for the relevant range by one. The 'Multisection Neuron Coverage' value is the ratio of the 'MNC Counter' value to the number of all neurons on the model.

### Interaction Metrics

**Sign-Sign Coverage:** This coverage criterion encompasses the sign states (positive, negative, or zero) of each component in the neural network inputs. This criterion aims to analyze the sensitivity of the network to signs. It can be measured by modifying the sign states of each component in the input data (positive, negative, or zero) and observing the outputs of the neural network.

**Value-Sign Coverage:** This coverage criterion encompasses the values and sign states of the inputs. It considers both the values and signs of each input component. This way, the neural network's response to both values and signs can be analyzed. It can be measured by modifying the values and sign states of the components in the input data and observing the outputs of the neural network.

**Sign-Value Coverage:** This coverage criterion encompasses the sign states and values of the inputs separately. Sign-Value Coverage allows for separate examination of the network's sensitivity to signs and values. It can be measured by modifying the sign states and values of the components in the input data and observing the outputs of the neural network.

**Value-Value Coverage:** This coverage criterion encompasses the values of the inputs. It targets different value ranges for each input component. This criterion aims to understand how the neural network behaves with different values. It can be measured by modifying the values of the components in the input data and observing the outputs of the neural network.

---

## Installation

First of all, Git system must be installed on the local machine in order to clone the project to the local machine via GitHub. After completing the necessary installation instructions suitable for the operating system from the [Git Download](https://git-scm.com/downloads) address and installing Git on the local machine, the project files will be cloned on the local machine by running the following command from the terminal.

```
git clone https://github.com/ESOGU-SRLAB/CleanAI.git
```

Then, you need to install Python on your device. Recommended Python version: > 3.8
For any operating operating system, you can download the Python installation file from this link or or you can follow the installation instructions and complete the installation steps. Python will be activated as a result of installing the file downloaded from this link on the local machine: [Python Download](https://www.python.org/downloads/)

In order for the library to run, the necessary dependencies must be installed. As a result of running the following command in the terminal on the project directory, the necessary dependencies will be installed automatically.

```
pip install -r requirements.txt
```

---

## Usage

### If you're dealing with a model on the local machine:

First, save your model that you have prepared using PyTorch. This model can be found in another directory and add the saved model to the main directory within the CleanAI project.

**_How to save a prepared model using PyTorch? (Ref: [PyTorch Tutorials](https://pytorch.org/tutorials/))_**

```python
torch.save(model, PATH)
```

The 'model' variable specifies your model object, and the 'PATH' variable specifies where you want to save the model.

Then copy the saved model to the CleanAI project main directory.

![Adding the model saved in the CleanAI project directory](https://i.ibb.co/92dRPtt/Screenshot-2023-06-20-112739.png "Adding the model saved in the CleanAI project directory")
**Figure 1:** Adding the model saved in the CleanAI project directory

In the next step, the class definition of the model must be given in the 'main.py' file and the model must be loaded. The process of giving the class definition of a sample model in 'main.py' and loading the model are shown below.

```python
class NeuralNetwork(nn.Module):
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
```

```python
model = NeuralNetwork()
model = torch.load("./test_model.pth")
```

Here, the string value passed as a parameter to the 'torch.load' function shows the directory where the model is located.

In the next stage, the preparation of the data set and transformer to be given to the model comes.

The image below shows the data set in the CleanAI project main directory, which contains the images that determine the inputs to be given to the model during the analysis of the model.

![Directory of test entries to be given to the model](https://i.ibb.co/FYSRNSC/Screenshot-2023-06-20-113935.png "Directory of test entries to be given to the model")
**Figure 2:** Directory of test entries to be given to the model

```python
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

image_loader = ImageLoader("./samples", transform)
```

At this stage, we would like to remind you that the definition of the variable named 'transform', which is passed as a parameter to the 'ImageLoader' class, will differ from model to model. For this, we recommend examining the documentation of the model in order to define the 'transform' variable suitable for the model to be tested and making the definition of 'transform' suitable for the model. It is important to do this step carefully in order to properly adjust the input data set to be given to the model during the analysis of the model.

In the next step, it is necessary to determine the parameters to be given to the CleanAI library. The 'how_many_samples' parameter will calculate how many inputs will be taken into account during the calculation of the average coverage values, the 'th_cov_val' parameter will be the threshold value during the calculation of the threshold coverage metric, the 'value_cov_th' expression will be the threshold value during the calculation of the value coverage metric, the 'top_k_val' parameter will be the top-k During the calculation of the value coverage value, the 'k' value indicates how much the 'node_intervals' value will be, which threshold value ranges will be taken into account during the calculation of the multisection neuron coverage metric, and the last parameter passed to the 'Analyzer' is whether the SS & SV & VS & VV metrics will be calculated.

```python
how_many_samples = 5
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
```

By calling the function named 'analyze' in 'Analyzer', the production of the analysis report is performed.

The final code in 'main.py' should be as follows. (Parameters should be adjusted according to the user.)

```python
class NeuralNetwork(nn.Module):
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

model = NeuralNetwork()
model = torch.load("./test_model.pth")

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
image_loader = ImageLoader("./samples", transform)

how_many_samples = 5
th_cov_val = 0.75
value_cov_th = 0.75
top_k_val = 3
node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

analyze = Analyzer(
    model,
    image_loader,
    how_many_samples,
    th_cov_val,
    value_cov_th,
    top_k_val,
    node_intervals,
)
analyzer.analyze()
```

Output values will be produced as a result of running the 'main.py' file.

```
python main.py
```

After the processes are completed, a file named 'Analysis\_[MODEL_NAME].pdf' will be created with analysis outputs.

![Report with analysis outputs](https://i.ibb.co/PNxrTwJ/Screenshot-2023-06-20-115928.png "Report with analysis outputs")
**Figure 3:** Report with analysis outputs

### If the model is to be loaded from a repository such as Torch Hub:

If the model is to be loaded from the Torch Hub rather than from the local directory, the following lines of code should be followed. The following code block shows how to load the ResNet18 model as an example.

```python
import torch

model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
```

The first parameter passed to the 'torch.hub.load' function indicates in which directory the model is located in the remote repository, the second parameter indicates how the model is named, and the 'pretrained' parameter indicates whether the model will use predetermined weights.

After the model is loaded, the data set of the model should be included in the project folder. The steps at this stage are the same as those described in the 'If you're dealing with a model on the local machine' topic.

The test input data of the model should be collected under a folder and placed in the project main directory.

![ResNet-18 dataset in project main directory](https://i.ibb.co/ynkLVKT/Screenshot-2023-07-31-173920.png "ResNet-18 dataset in project main directory")
**Figure 4:** ResNet-18 dataset in project main directory

As seen in the image above, which test inputs you want to work with during the analysis of the model, these test inputs should be collected under a folder (in the example: 'resnet18_dataset' folder) as images with .png, .jpg or .jpeg extensions. NOTE: If there are extra folders/nested folders in the data set folder, all folders will be scanned and all the images in it will be added to the data set to be used.

```python

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to desired size
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
])

image_loader = ImageLoader("./resnet18_dataset", transform)
```

At this stage, again, we would like to remind you that the definition of the variable named 'transform', which is passed as a parameter to the 'ImageLoader' class, will differ from model to model. For this, we recommend examining the documentation of the model in order to define the 'transform' variable suitable for the model to be tested and making the definition of 'transform' suitable for the model. It is important to do this step carefully in order to properly adjust the input data set to be given to the model during the analysis of the model.

```python
import torch

model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True) # Load model from hub

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to desired size
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
])

image_loader = ImageLoader("./resnet18_dataset", transform) # Load dataset to be used in analysis phase

# Define the parameters
how_many_samples = 5
th_cov_val = 0.75
value_cov_th = 0.75
top_k_val = 3
node_intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]

analyze = Analyzer(
    model,
    image_loader,
    how_many_samples,
    th_cov_val,
    value_cov_th,
    top_k_val,
    node_intervals,
)
analyzer.analyze()
```

Above is an example code block of how the 'main' function will look. There is a sample code block on how to load the model from the hub, how to define the 'transform' variable, how to load the data set in the project main directory, how to determine the parameters to be used during the analysis (which parameter is used for what in the upper section), and how to start the analysis process.

After all these processes, a file with the same name as your model (Analysis\_[ModelName].pdf) in which the output values are saved will be produced in the project main directory.

![Report with analysis outputs](https://i.ibb.co/PNxrTwJ/Screenshot-2023-06-20-115928.png "Report with analysis outputs")
**Figure 5:** Report with analysis outputs

---

## Sample Analysis Reports (Case Study and Output Results on Different ResNet Models)

As an example, in this study, images in the datasets/resnet_50samples directory were used as input. These input images were fed to different ResNet models and detailed output reports were obtained for 11 coverage metrics.

<div style="display: flex; justify-content: center;">
    <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
        <img src="https://i.ibb.co/dpvMnHt/n01440764-ILSVRC2012-val-00009111.jpg" alt="Sample image-1 used in the study" style="width: 45%; margin-bottom: 10px;" />
        <p><strong>Figure 6:</strong> Sample image-1 used in the study</p>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin-left: 20px;">
        <img src="https://i.ibb.co/xq8115j/n01443537-ILSVRC2012-val-00000994.jpg" alt="Sample image-2 used in the study" style="width: 45%; margin-bottom: 10px;" />
        <p><strong>Figure 7:</strong> Sample image-2 used in the study</p>
    </div>
</div>

![Input images used to obtain sample analysis reports (50, taken from ImageNet)](https://i.ibb.co/P4dHtY8/50-test-inputs.jpg "Input images used to obtain sample analysis reports (50, taken from ImageNet)")
**Figure 8:** Input images used to obtain sample analysis reports (50, taken from ImageNet)

The figure above shows the images used to obtain analysis reports and given as test input to the model.

**Tablo 1:** Total Number of Parameters and Layers of ResNet Models

| Model      | Total Parameter | Number of Layers |
| ---------- | --------------- | ---------------- |
| ResNet-18  | 11,689,512      | 68               |
| ResNet-34  | 21,797,672      | 116              |
| ResNet-50  | 25,557,032      | 151              |
| ResNet-101 | 44,549,160      | 287              |
| ResNet-152 | 60,192,808      | 423              |

Table 1 shows the number of parameters available in different ResNet versions.

**Tablo 2:** Neuron Coverage Values of ResNet Models (Input Used: Figure 6)

| Model  | Number of Neurons Covered | Total Number of Neurons | Coverage Value |
| ------ | ------------------------- | ----------------------- | -------------- |
| RN-18  | 646,108                   | 1,555,456               | 41.54%         |
| RN-34  | 879,543                   | 2,182,656               | 40.30%         |
| RN-50  | 2,234,998                 | 6,322,176               | 35.35%         |
| RN-101 | 3,326,557                 | 9,734,144               | 34.17%         |
| RN-152 | 4,810,465                 | 13,948,928              | 34.49%         |

Table 2 shows the Neuron Coverage Metric values ​​of different ResNet versions. Figure 6 is used as an example input here.

**Tablo 3:** Neuron Coverage Values of ResNet Models (Values Created by Selecting 50 Random Inputs, Figure 8)

| Model  | Number of Neurons Covered | Total Number of Neurons | Coverage Value |
| ------ | ------------------------- | ----------------------- | -------------- |
| RN-18  | 31,750,758                | 77,772,800              | 40.83%         |
| RN-34  | 43,559,939                | 109,132,800             | 39.91%         |
| RN-50  | 110,228,414               | 316,108,800             | 34.87%         |
| RN-101 | 164,429,818               | 486,707,200             | 33.78%         |
| RN-152 | 240,223,749               | 697,446,400             | 34.44%         |

Table 3 shows the Neuron Coverage Metric values ​​of different ResNet versions. All images in Figure 8 is used as an example input here.

**Tablo 4:** Threshold Coverage Values of ResNet Models (Threshold Coverage Value = 0.75, Input Used: Figure 6)

| Model  | Number of Neurons Covered | Total Number of Neurons | Coverage Value |
| ------ | ------------------------- | ----------------------- | -------------- |
| RN-18  | 157,373                   | 1,555,456               | 10.12%         |
| RN-34  | 290,542                   | 2,182,656               | 13.31%         |
| RN-50  | 131,956                   | 6,322,176               | 2.09%          |
| RN-101 | 171,694                   | 9,734,144               | 1.76%          |

Table 4 shows the Threshold Coverage Metric values ​​of different ResNet versions. Figure 6 is used as an example input here.

**Tablo 5:** Neuron Boundary Coverage Values of ResNet Models (Test Entry: Figure 6)

| Model      | Number of Neurons Exceeding the Boundary | Total Number of Neurons | NB Coverage |
| ---------- | ---------------------------------------- | ----------------------- | ----------- |
| ResNet-18  | 0                                        | 1,555,456               | 0.00%       |
| ResNet-34  | 0                                        | 2,182,656               | 0.00%       |
| ResNet-50  | 0                                        | 6,322,176               | 0.00%       |
| ResNet-101 | 0                                        | 9,734,144               | 0.00%       |
| ResNet-152 | 0                                        | 13,948,928              | 0.00%       |

Table 5 shows the Neuron Boundary Coverage Metric values ​​of different ResNet versions. Figure 6 is used as an example input here.

**Tablo 6:** Top-K Neuron Coverage Values of ResNet Models (Test Entry: Figure 6, K Value = 3)

| Model  | TKNC Sum | Number of Selected Neurons | Top-K Neuron Average |
| ------ | -------- | -------------------------- | -------------------- |
| RN-18  | 126.99   | 27                         | 4.70                 |
| RN-34  | 263.57   | 51                         | 5.17                 |
| RN-50  | 163.58   | 51                         | 3.21                 |
| RN-101 | 309.14   | 102                        | 3.03                 |
| RN-152 | 467.94   | 153                        | 3.06                 |

Table 6 shows the Top-K Neuron Coverage Metric values ​​of different ResNet versions. Figure 6 is used as an example input here.

**Tablo 7:** Multisection Neuron Coverage Values of ResNet Models (Test Entry: Figure 6)

| Model  | Threshold Ranges | Number of Neurons Covered (MNC Counter) | Total Number of Neurons | Multisection Neuron Coverage |
| ------ | ---------------- | --------------------------------------- | ----------------------- | ---------------------------- |
| RN-18  | 0 - 0.1          | 616,067                                 | 1,555,456               | 39.61%                       |
|        | 0.1 - 0.2        | 146,704                                 | 1,555,456               | 9.43%                        |
|        | 0.2 - 0.3        | 165,872                                 | 1,555,456               | 10.66%                       |
|        | 0.3 - 0.4        | 157,284                                 | 1,555,456               | 10.11%                       |
|        | 0.4 - 0.5        | 123,655                                 | 1,555,456               | 7.95%                        |
| RN-34  | 0 - 0.1          | 843,695                                 | 2,182,656               | 38.65%                       |
|        | 0.1 - 0.2        | 192,149                                 | 2,182,656               | 8.80%                        |
|        | 0.2 - 0.3        | 212,462                                 | 2,182,656               | 9.73%                        |
|        | 0.3 - 0.4        | 199,360                                 | 2,182,656               | 9.13%                        |
|        | 0.4 - 0.5        | 160,711                                 | 2,182,656               | 7.36%                        |
| RN-50  | 0 - 0.1          | 3,648,786                               | 6,322,176               | 57.71%                       |
|        | 0.1 - 0.2        | 897,189                                 | 6,322,176               | 14.19%                       |
|        | 0.2 - 0.3        | 702,219                                 | 6,322,176               | 11.11%                       |
|        | 0.3 - 0.4        | 431,215                                 | 6,322,176               | 6.82%                        |
|        | 0.4 - 0.5        | 257,773                                 | 6,322,176               | 4.08%                        |
| RN-101 | 0 - 0.1          | 6,031,889                               | 9,734,144               | 61.97%                       |
|        | 0.1 - 0.2        | 1,347,685                               | 9,734,144               | 13.84%                       |
|        | 0.2 - 0.3        | 969,030                                 | 9,734,144               | 9.95%                        |
|        | 0.3 - 0.4        | 563,331                                 | 9,734,144               | 5.79%                        |
|        | 0.4 - 0.5        | 310,601                                 | 9,734,144               | 3.19%                        |
| RN-152 | 0 - 0.1          | 8,243,615                               | 13,948,928              | 59.10%                       |
|        | 0.1 - 0.2        | 1,910,427                               | 13,948,928              | 13.70%                       |
|        | 0.2 - 0.3        | 1,409,205                               | 13,948,928              | 10.10%                       |
|        | 0.3 - 0.4        | 912,240                                 | 13,948,928              | 6.54%                        |
|        | 0.4 - 0.5        | 554,407                                 | 13,948,928              | 3.97%                        |

Table 7 shows the Neuron Coverage Metric values ​​of different ResNet versions. Figure 6 is used as an example input here.

**Tablo 8:** Sign Coverage and Value Coverage Values of ResNet Models (Test Inputs: Figure 6 and Figure 7, Threshold Value for Value Coverage = 0.75)

| Model  | Coverage Metric | Number of Neurons Covered | Total Number of Neurons | Coverage Value |
| ------ | --------------- | ------------------------- | ----------------------- | -------------- |
| RN-18  | Sign Coverage   | 700,782                   | 1,555,456               | 45.05%         |
|        | Value Coverage  | 119,449                   | 1,555,456               | 7.68%          |
| RN-34  | Sign Coverage   | 928,468                   | 2,182,656               | 42.54%         |
|        | Value Coverage  | 209,706                   | 2,182,656               | 9.61%          |
| RN-50  | Sign Coverage   | 3,455,509                 | 6,322,176               | 54.66%         |
|        | Value Coverage  | 133,208                   | 6,322,176               | 2.11%          |
| RN-101 | Sign Coverage   | 4,948,698                 | 9,734,144               | 50.84%         |
|        | Value Coverage  | 177,566                   | 9,734,144               | 1.82%          |
| RN-152 | Sign Coverage   | 6,823,384                 | 13,948,928              | 48.92%         |
|        | Value Coverage  | 281,901                   | 13,948,928              | 2.02%          |

Table 8 shows the Sign Coverage and Value Coverage Metric values ​​of different ResNet versions. Figure 6 and Figure 7 are used as an example input here.

**Tablo 9:** File Sizes, Accuracy Values and Neuron Coverage Values of the Models Obtained as a Result of Running ResNet Models Under the ImageNet Dataset (Input Used: Figure 3)

| Model      | File Size of Weights (MB) | Accuracy | Neuron Coverage |
| ---------- | ------------------------- | -------- | --------------- |
| ResNet-18  | 44.7                      | 84.00%   | 41.54%          |
| ResNet-34  | 83.3                      | 86.00%   | 40.30%          |
| ResNet-50  | 97.8                      | 90.00%   | 35.35%          |
| ResNet-101 | 170.5                     | 88.00%   | 34.17%          |
| ResNet-152 | 230.4                     | 88.00%   | 34.49%          |

Table 9 provides summary information about the different ResNet versions.

---

## Collobrators

You can access the social media accounts of our teammates who supported the development phase from the links below.

- Osman Çağlar <a target="_blank" href="https://www.linkedin.com/in/osmancaglar/" rel="nofollow noreferrer">
  <img src="https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="linkedin">
  </a> &nbsp;
- Abdul Hannan Ayubi <a target="_blank" href="https://www.linkedin.com/in/abdulhannanayubi/" rel="nofollow noreferrer">
  <img src="https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="linkedin">
  </a> &nbsp;
- Furkan Taşkın <a target="_blank" href="https://www.linkedin.com/in/furkan-taskin/" rel="nofollow noreferrer">
  <img src="https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="linkedin">
  </a> &nbsp;
- Sergen Aşık <a target="_blank" href="https://www.linkedin.com/in/sergenasik/" rel="nofollow noreferrer">
  <img src="https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="linkedin">
  </a> &nbsp;
- Cem Bağlum <a target="_blank" href="https://www.linkedin.com/in/cembaglum/" rel="nofollow noreferrer">
  <img src="https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="linkedin">
  </a> &nbsp;
- Dr. Uğur Yayan <a target="_blank" href="https://www.linkedin.com/in/uguryayan/" rel="nofollow noreferrer">
  <img src="https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="linkedin">
  </a> &nbsp;

---

## Credits

<div style="display: flex; justify-content: center;">
    <img src="https://tubitak.gov.tr/sites/default/files/2023-08/logo.svg" alt="TÜBİTAK Logo" style="width: 100px; margin-left: auto; margin-right: auto; margin-bottom: 15px" />
</div>

**This project was supported by TÜBİTAK within the scope of '2209-B Undergraduate Research Projects Support Program for Industry (2209-B Sanayiye Yönelik Lisans Araştırma Projeleri Destekleme Programı)'.**

---

## Licence

---
