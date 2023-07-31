# CleanAI - Evaluation of Deep Neural Network Quality by CleanAI Coverage Metrics Library
![lang](https://img.shields.io/github/languages/top/inomuh/imfit)

## Contents

The document includes the following titles:

- [What is CleanAI](#whatiscleanai)
- [Coverage Metrics](#coveragemetrics)
- [Installation](#installation)
- [Usage](#usage)
- [Collobrators](#collobrators)
- [Credits](#credits)
- [License](#license)

## What is CleanAI?
CleanAI is a white-box testing library that utilizes coverage metrics to evaluate the structural analysis, quality, and reliability parameters of DNN models. The library incorporates eleven coverage test methods and enables developers to perform essential model analyses and generate output reports. By using CleanAI, developers can effectively evaluate the quality of DNNs and make informed decisions.

CleanAI is dedicated to enhancing the quality assurance of AI systems through effective testing methodologies and valuable insights for developers. Our research aims to advance AI safety, promote responsible development and deployment, and mitigate risks associated with AI technologies. By improving reliability and quality assurance, we can foster trust and ensure the ethical and effective use of AI across domains, ultimately enhancing its societal impact. The utilization of our CleanAI coverage metrics library aims to enhance reliability while addressing ethical concerns. Through comprehensive testing and analysis, we aim to minimize risks and improve decision-making, reliability, explainability, and accountability in AI systems, thus enhancing AI safety.

Within the scope of this study, our CleanAI library measures various coverage metrics to determine the quality of deep neural networks. These coverage metrics include **Neuron Coverage**, **Threshold Coverage**, **Sign Coverage**, **Value Coverage**, **Sign-Sign Coverage**, **Value-Sign Coverage**, **Sign-Value Coverage**, **Value-Value Coverage**, **Neuron Boundary Coverage**, **Multisection Neuron Coverage**, and **Top K Neuron Coverage**.

## Coverage Metrics

**Neuron Coverage:** This coverage criterion tracks the activation status of each neuron in a neural network. It aims to determine which neurons are activated and which ones are not. In our implementation of the Neuron Coverage criterion, we have made a difference compared to the literature. The distinction lies in how we calculate the activation status of neurons. Instead of directly comparing the after values of each neuron with a fixed threshold, we take a different approach. After each activation function, we compute the average of the resulting "after value" for each layer. Then, we check if the after value of each neuron is greater than the calculated average for its corresponding layer. Neurons with after values greater than the average are considered active. By calculating the average of all neurons in each layer and comparing individual neuron values to their respective average, we determine the activation status of each neuron. This approach allows us to assess the proportion of activated neurons in the neural network based on the after values relative to the layer averages.

**Threshold Coverage:** The threshold coverage value of a model is the ratio of the number of covered neurons (number of neurons greater than the threshold value) to the total number of neurons in the model.

**Neuron Boundary Coverage:** Neuron Boundary Coverage (NBC) is a metric used to evaluate the coverage of decision boundaries in a deep neural network (DNN). It measures the percentage of decision boundaries in the network that have been activated or crossed by the input samples. **_How is it calculated?_** NBC receives a random set of inputs from the user, and as a result of these inputs, it determines the maximum and minimum interval value for each layer. Then, for the input data to be checked, it is checked whether each neuron belonging to each layer is within the maximum and minimum range of this layer. If it is within this range, the 'NBC Counter' value is increased by one.

**Top-K Neuron Coverage:** Top-K Neuron Coverage (TKNC) is a metric used to evaluate the activation patterns and coverage of neurons in a deep neural network (DNN). It measures the percentage of neurons that are activated for a given set of input samples. The idea behind TKNC is to assess how well a set of input samples can activate different neurons in the network. **_How is it calculated?_** TKNC travels through all layers on a model one by one and ranks the neuron values of each layer in order from largest to smallest. Then it takes k neurons in each layer and adds it to a list. It then creates a value called 'TKNC Sum', which represents the sum of neurons in this list. The 'Number of Selected Neurons' value shows how many neurons were selected on the whole model as a result of k neurons from each layer. The 'Mean of Top-K Neurons' value shows the ratio of the 'TKNC Sum' value to the 'Number of Selected Neurons' value.

**Multisection Neuron Coverage:** Multisection Neuron Coverage (MNC) specifically focuses on assessing the coverage of individual neurons within the model. The goal of MNC is to evaluate the degree to which the decisions made by individual neurons have been exercised by the test cases. It helps identify potential shortcomings in the model's behavior and reveal areas that may require further testing. It provides the user with the information of how many neurons are found according to the threshold value ranges given by the user. **_How is it calculated?_** The MNC receives threshold ranges from the user. Then, it evaluates all the neurons on the model and checks whether each neuron is within these threshold ranges. If the corresponding neuron is within this threshold value, it increases the 'MNC Counter' value found for the relevant range by one. The 'Multisection Neuron Coverage' value is the ratio of the 'MNC Counter' value to the number of all neurons on the model.

**Sign Coverage:** When given two different test inputs, it checks whether the signs of a specific neuron's value after the activation function are the same. If the signs are not the same, the counter is incremented.

**Value Coverage:** When given two different test inputs, it checks whether the difference between the values of a specific neuron after the activation function is greater than the given threshold value. If the difference is greater than the threshold value, the counter is incremented.

**Sign-Sign Coverage:** This coverage criterion encompasses the sign states (positive, negative, or zero) of each component in the neural network inputs. This criterion aims to analyze the sensitivity of the network to signs. It can be measured by modifying the sign states of each component in the input data (positive, negative, or zero) and observing the outputs of the neural network.

**Value-Sign Coverage:** This coverage criterion encompasses the values and sign states of the inputs. It considers both the values and signs of each input component. This way, the neural network's response to both values and signs can be analyzed. It can be measured by modifying the values and sign states of the components in the input data and observing the outputs of the neural network.

**Sign-Value Coverage:** This coverage criterion encompasses the sign states and values of the inputs separately. Sign-Value Coverage allows for separate examination of the network's sensitivity to signs and values. It can be measured by modifying the sign states and values of the components in the input data and observing the outputs of the neural network.

**Value-Value Coverage:** This coverage criterion encompasses the values of the inputs. It targets different value ranges for each input component. This criterion aims to understand how the neural network behaves with different values. It can be measured by modifying the values of the components in the input data and observing the outputs of the neural network.

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

After the processes are completed, a file named 'Analysis_[MODEL_NAME].pdf' will be created with analysis outputs.

![Report with analysis outputs](https://i.ibb.co/PNxrTwJ/Screenshot-2023-06-20-115928.png "Report with analysis outputs")

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

After all these processes, a file with the same name as your model (Analysis_[ModelName].pdf) in which the output values are saved will be produced in the project main directory.

![Report with analysis outputs](https://i.ibb.co/PNxrTwJ/Screenshot-2023-06-20-115928.png "Report with analysis outputs")


## Collobrators
You can access the social media accounts of our teammates who supported the development phase from the links below.

- Osman Çağlar <a href="https://www.linkedin.com/in/osmancaglar/" rel="nofollow noreferrer">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="linkedin">
  </a> &nbsp;
- Abdul Hannan Ayubi <a href="https://www.linkedin.com/in/abdulhannanayubi/" rel="nofollow noreferrer">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="linkedin">
  </a> &nbsp;
- Furkan Taşkın <a href="https://www.linkedin.com/in/furkan-taskin/" rel="nofollow noreferrer">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="linkedin">
  </a> &nbsp;
- Sergen Aşık <a href="https://www.linkedin.com/in/sergenasik/" rel="nofollow noreferrer">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="linkedin">
  </a> &nbsp;
- Cem Bağlum <a href="https://www.linkedin.com/in/cembaglum/" rel="nofollow noreferrer">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="linkedin">
  </a> &nbsp;
- Dr. Uğur Yayan <a href="https://www.linkedin.com/in/uguryayan/" rel="nofollow noreferrer">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="linkedin">
  </a> &nbsp; 

## Credits

## Licence

