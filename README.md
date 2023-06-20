# CleanAI - Evaluation of Deep Neural Network Quality by CleanAI Coverage Metrics Library

## Contents

The document includes the following titles:

- [What is CleanAI](#whatiscleanai)
- [Coverage Metrics](#coveragemetrics)
- [Installation](#installation)
- [Usage](#usage)
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
In order for the library to run, the necessary dependencies must be installed.
```
pip install -r requirements.txt
```

## Usage
First, save your model that you have prepared using PyTorch. This model can be found in another directory and add the saved model to the main directory within the CleanAI project.

**_How to save a prepared model using PyTorch? (Ref: PyTorch Tutorial)_**
```
torch.save(model, PATH)
```
The 'model' variable specifies your model object, and the 'PATH' variable specifies where you want to save the model.

Then copy the saved model to the CleanAI project main directory.

![Adding the model saved in the CleanAI project directory](https://i.ibb.co/92dRPtt/Screenshot-2023-06-20-112739.png "Adding the model saved in the CleanAI project directory")

In the next step, the class definition of the model must be given in the 'main.py' file and the model must be loaded. The process of giving the class definition of a sample model in 'main.py' and loading the model are shown below.



```
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

```
model = NeuralNetwork()
model = torch.load("./test_model.pth")
```

In the next stage, the preparation of the data set and transformer to be given to the model comes.

The image below shows the data set in the CleanAI project main directory, which contains the images that determine the inputs to be given to the model during the analysis of the model.

![Directory of test entries to be given to the model](https://i.ibb.co/FYSRNSC/Screenshot-2023-06-20-113935.png "Directory of test entries to be given to the model")


```
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
    
image_loader = ImageLoader("./samples", transform)
```

In the next step, it is necessary to determine the parameters to be given to the CleanAI library. The 'how_many_samples' parameter will calculate how many inputs will be taken into account during the calculation of the average coverage values, the 'th_cov_val' parameter will be the threshold value during the calculation of the threshold coverage metric, the 'value_cov_th' expression will be the threshold value during the calculation of the value coverage metric, the 'top_k_val' parameter will be the top-k During the calculation of the value coverage value, the 'k' value indicates how much the 'node_intervals' value will be, which threshold value ranges will be taken into account during the calculation of the multisection neuron coverage metric, and the last parameter passed to the 'Analyzer' is whether the SS & SV & VS & VV metrics will be calculated.

```
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

```
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

After the processes are completed, a file named 'Analysis_[MODEL_NAME].pdf' will be created with analysis outputs.

![Report with analysis outputs](https://i.ibb.co/PNxrTwJ/Screenshot-2023-06-20-115928.png "Report with analysis outputs")


