import os
import os.path
from dataset import FashionMNISTLoader
from model import NeuralNetwork
from train_model import NeuralNetworkTrainer
from neural_network_profiler import NeuralNetworkProfiler
from neuron_coverage import NeuronCoverage
from model_architecture_utils import ModelArchitectureUtils

def main():
    model_name = 'model.pth'
    if(os.path.isfile('./' + model_name)):
        print('Model already exists')
        loaded_trainer = NeuralNetworkTrainer.load_model(model_name)
        model = loaded_trainer.model
        #print(model)
    else:
        print('Training model')
        model = NeuralNetwork()
        trainer = NeuralNetworkTrainer(model)
        trainer.train_and_save(FashionMNISTLoader.train_dataloader, FashionMNISTLoader.test_dataloader, learning_rate=1e-3, batch_size=64, epochs=5, model_name='model.pth')
        #print(model)

    random_inputs_with_labels = FashionMNISTLoader.get_random_inputs_with_labels(5)
    random_inputs = FashionMNISTLoader.get_random_inputs(5)

    #print(random_inputs)

    #NeuralNetwork.make_prediction(model, random_inputs_with_labels)

    model_layers_arr = NeuralNetworkProfiler.get_layer_names(model)
    #print(model_layers_arr)
    #NeuralNetworkProfiler.run_model_with_array(random_inputs[0], model_layers_arr)

    model_architecture = NeuralNetworkProfiler.get_model_architecture_dict_of_input(random_inputs[0], model_layers_arr)
    after_values = ModelArchitectureUtils.get_after_values_for_specific_layer(model_architecture, 0)
    num_of_covered_neurons, total_neurons, neuron_coverage = NeuronCoverage.get_neuron_coverage_for_single_layer(after_values)
    print(f"Num of covered neurons: {num_of_covered_neurons}\nNum of total neurons: {total_neurons}\nNeuron coverage: {neuron_coverage}")

    after_values_all_layers = ModelArchitectureUtils.get_after_values_for_all_layers(model_architecture)
    #print(after_values_all_layers)
    num_of_covered_neurons, total_neurons, neuron_coverage = NeuronCoverage.get_neuron_coverage_for_all_layers(after_values_all_layers)
    print(f"Num of covered neurons: {num_of_covered_neurons}\nNum of total neurons: {total_neurons}\nNeuron coverage: {neuron_coverage}")

    num_of_covered_neurons, total_neurons, neuron_coverage = NeuronCoverage.get_threshold_coverage_for_all_layers(after_values_all_layers)
    print(f"TH Num of covered neurons: {num_of_covered_neurons}\nTH Num of total neurons: {total_neurons}\nTH Neuron coverage: {neuron_coverage}")

    #NeuralNetworkProfiler.visualize_model(model, random_inputs[0])




if __name__ == '__main__':
    main()
