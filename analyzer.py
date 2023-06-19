import torch
import torch.nn as nn

from driver import Driver
from print_utils import PDFWriter
from image_loader import ImageLoader


class Analyzer:
    def __init__(
        self,
        model,
        image_loader,
        how_many_samples,
        th_cov_val,
        value_cov_th,
        top_k_val,
        node_intervals,
        ss_sv_vv_vs_cov=False,
    ):
        self.model = model
        self.image_loader = image_loader
        self.driver = Driver(self.model)

        self.th_cov_val = th_cov_val
        self.value_cov_th = value_cov_th
        self.top_k_val = top_k_val
        self.node_intervals = node_intervals
        self.ss_sv_vv_vs_cov = ss_sv_vv_vs_cov

        self.model_info = self.driver.get_model_informations()

        self.pdf_file_name = f"Analysis_{self.model_info['name']}.pdf"
        self.pdf_writer = PDFWriter(self.pdf_file_name)

        self.sample, self.random_image_name = self.image_loader.get_random_input()
        self.sample_II, self.random_image_name_II = image_loader.get_random_input()
        self.samples = []
        for i in range(how_many_samples):
            sample, random_image_name = self.image_loader.get_random_input()
            self.samples.append(sample)

    def analyze(self):
        self.pdf_writer.add_text("CleanAI Analysis Report", font_size=20, is_bold=True)

        # Save model informations to PDF
        data = [
            ["", "Model name", "Total params", "Number of layers"],
            [
                "Informations",
                self.model_info["name"],
                str(self.model_info["total_params"]),
                str(self.model_info["num_layers"]),
            ],
        ]

        self.pdf_writer.add_text("Model Informations", font_size=16, is_bold=True)
        self.pdf_writer.add_text(
            "The table below shows general information about the '"
            + self.model_info["name"]
            + "' model.",
            font_size=14,
            is_bold=False,
        )
        self.pdf_writer.add_space(5)
        self.pdf_writer.add_table(data)
        self.pdf_writer.add_space(20)

        # Calculation of neuron coverage of all layers of the model
        data = [
            [
                "Layer Index",
                "Activation Function",
                "Number of Covered Neurons",
                "Number of Total Neurons",
                "Coverage Value",
                "Mean of Layer",
            ]
        ]

        coverage_values_of_layers = self.driver.get_coverage_of_layers(self.sample)
        self.pdf_writer.add_text(
            "Coverage Values of Layers (For Only One Input)",
            font_size=16,
            is_bold=True,
        )
        self.pdf_writer.add_text(
            f"The table below shows coverage values about the '"
            + self.model_info["name"]
            + "' model's all layers. The 'mean of layer' value shows the average of neurons in that layer. When calculating the number of covered neurons, this value is accepted as the threshold value for that layer. NOTE: The coverage value of a layer is the ratio of the number of covered neurons to the total number of neurons in that layer. The values in the table below, it was formed as a result of giving the '"
            + self.random_image_name
            + "' input in the data set to the model.",
            font_size=14,
            is_bold=False,
        )
        self.pdf_writer.add_space(5)

        for idx, (
            layer_index,
            activation_fn,
            mean_of_layer,
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) in enumerate(coverage_values_of_layers):
            data.append(
                [
                    "Layer " + str(layer_index),
                    str(activation_fn),
                    str(num_of_covered_neurons),
                    str(total_neurons),
                    f"{coverage * 100:.2f}%",
                    f"{mean_of_layer:.2f}",
                ]
            )

        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = self.driver.get_coverage_of_model(self.sample)
        data.append(
            [
                "All model",
                "-",
                str(num_of_covered_neurons),
                str(total_neurons),
                f"{coverage * 100:.2f}%",
            ]
        )
        self.pdf_writer.add_table(data)
        self.pdf_writer.add_space(20)

        # Calculation of neuron coverage of all layers of the model (multiple inputs)

        data = [
            [
                "Layer index",
                "Number of covered neurons",
                "Number of total neurons",
                "Coverage value",
            ]
        ]
        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = self.driver.get_average_coverage_of_model(self.samples)
        self.pdf_writer.add_text(
            "Coverage Values of Layers (For Multiple Inputs) "
            + str(len(self.samples))
            + " Inputs",
            font_size=16,
            is_bold=True,
        )
        self.pdf_writer.add_text(
            f"The table below shows coverage values for multiple inputs about the '"
            + self.model_info["name"]
            + "' model. The values in the table below, it was formed as a result of giving the '"
            + str(len(self.samples))
            + "' inputs in the data set to the model.",
            font_size=14,
            is_bold=False,
        )
        self.pdf_writer.add_space(5)
        data.append(
            [
                "All model",
                str(num_of_covered_neurons),
                str(total_neurons),
                f"{coverage * 100:.2f}%",
            ]
        )
        self.pdf_writer.add_table(data)
        self.pdf_writer.add_space(20)

        # Calculation of threshold coverage of all layers of the model (multiple inputs)

        data = [
            [
                "Layer index",
                "Activation function",
                "Number of covered neurons",
                "Number of total neurons",
                "Coverage value",
            ]
        ]

        self.pdf_writer.add_text(
            "Threshold Coverage Values of Layers (TH = " + str(self.th_cov_val) + ")",
            font_size=16,
            is_bold=True,
        )
        self.pdf_writer.add_text(
            f"The table below shows threshold coverage values about the '"
            + self.model_info["name"]
            + "' model's all layers. NOTE: The threshold coverage value of a layer is the ratio of the number of covered neurons (number of neurons greater than the threshold value) to the total number of neurons in that layer. The values in the table below, it was formed as a result of giving the '"
            + self.random_image_name
            + "' input in the data set to the model.",
            font_size=14,
            is_bold=False,
        )
        self.pdf_writer.add_space(5)

        coverage_values_of_layers = self.driver.get_th_coverage_of_layers(
            self.sample, self.th_cov_val
        )
        for idx, (
            layer_index,
            activation_fn,
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) in enumerate(coverage_values_of_layers):
            data.append(
                [
                    "Layer " + str(layer_index),
                    str(activation_fn),
                    str(num_of_covered_neurons),
                    str(total_neurons),
                    f"{coverage * 100:.2f}%",
                ]
            )

        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = self.driver.get_th_coverage_of_model(self.sample, self.th_cov_val)
        data.append(
            [
                "All model",
                "-",
                str(num_of_covered_neurons),
                str(total_neurons),
                f"{coverage * 100:.2f}%",
            ]
        )
        self.pdf_writer.add_table(data)
        self.pdf_writer.add_space(20)

        # Calculation of value coverage & sign coverage of the model

        data = [
            [
                "Coverage Metric",
                "Number of covered neurons",
                "Number of total neurons",
                "Coverage value",
            ]
        ]

        self.pdf_writer.add_text(
            "Sign Coverage and Value Coverage (TH = "
            + str(self.value_cov_th)
            + ") Values of Model",
            font_size=16,
            is_bold=True,
        )
        self.pdf_writer.add_text(
            f"The table below shows Sign Coverage and Value Coverage values of the '"
            + self.model_info["name"]
            + "' model. Sign Coverage: When given two different test inputs, it checks whether the signs of a specific neuron's value after the activation function are the same. If the signs are not the same, the counter is incremented. Value Coverage: When given two different test inputs, it checks whether the difference between the values of a specific neuron after the activation function is greater than the given threshold value. If the difference is greater than the threshold value, the counter is incremented. The values in the table below, it was formed as a result of giving the '"
            + self.random_image_name
            + "' and '"
            + self.random_image_name_II
            + "' input in the data set to the model.",
            font_size=14,
            is_bold=False,
        )
        self.pdf_writer.add_space(5)

        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = self.driver.get_sign_coverage_of_model(self.sample, self.sample_II)
        data.append(
            [
                "Sign Coverage",
                str(num_of_covered_neurons),
                str(total_neurons),
                f"{coverage * 100:.2f}%",
            ]
        )

        (
            num_of_covered_neurons,
            total_neurons,
            coverage,
        ) = self.driver.get_value_coverage_of_model(
            self.sample, self.sample_II, self.value_cov_th
        )
        data.append(
            [
                "Value Coverage",
                str(num_of_covered_neurons),
                str(total_neurons),
                f"{coverage * 100:.2f}%",
            ]
        )

        self.pdf_writer.add_space(5)
        self.pdf_writer.add_table(data)

        self.pdf_writer.add_space(20)

        # Calculation of SS & SV & VS && VV of the model
        if self.ss_sv_vv_vs_cov:
            data = [
                [
                    "Coverage Metric",
                    "Number of covered neuron pairs",
                    "Number of total neuron pairs",
                    "Coverage value",
                ]
            ]

            self.pdf_writer.add_text(
                "SS, SV, VS and VV Coverage (TH = "
                + str(self.value_cov_th)
                + ") Values of Model",
                font_size=16,
                is_bold=True,
            )
            self.pdf_writer.add_text(
                f"The table below shows Sign-Sign Coverage, Sign-Value Coverage, Value-Sign Coverage and Value-Value Coverage values of the '"
                + self.model_info["name"]
                + "' model. Sign-Sign Coverage: When given two different test inputs, it checks whether the signs of a specific neuron's value after the activation function are the same. If the signs are not the same, the counter is incremented. Value Coverage: When given two different test inputs, it checks whether the difference between the values of a specific neuron after the activation function is greater than the given threshold value. If the difference is greater than the threshold value, the counter is incremented. The values in the table below, it was formed as a result of giving the '"
                + self.random_image_name
                + "' and '"
                + self.random_image_name_II
                + "' input in the data set to the model.",
                font_size=14,
                is_bold=False,
            )
            self.pdf_writer.add_space(5)

            (
                num_of_covered_neurons,
                total_neurons,
                coverage,
            ) = self.driver.get_ss_coverage_of_model(self.sample, self.sample_II)
            data.append(
                [
                    "Sign-Sign Coverage",
                    str(num_of_covered_neurons),
                    str(total_neurons),
                    f"{coverage * 100:.2f}%",
                ]
            )

            (
                num_of_covered_neurons,
                total_neurons,
                coverage,
            ) = self.driver.get_sv_coverage_of_model(
                self.sample, self.sample_II, self.value_cov_th
            )
            data.append(
                [
                    "Sign-Value Coverage",
                    str(num_of_covered_neurons),
                    str(total_neurons),
                    f"{coverage * 100:.2f}%",
                ]
            )

            (
                num_of_covered_neurons,
                total_neurons,
                coverage,
            ) = self.driver.get_vs_coverage_of_model(
                self.sample, self.sample_II, self.value_cov_th
            )
            data.append(
                [
                    "Value-Sign Coverage",
                    str(num_of_covered_neurons),
                    str(total_neurons),
                    f"{coverage * 100:.2f}%",
                ]
            )

            (
                num_of_covered_neurons,
                total_neurons,
                coverage,
            ) = self.driver.get_vv_coverage_of_model(
                self.sample, self.sample_II, self.value_cov_th
            )
            data.append(
                [
                    "Value-Value Coverage",
                    str(num_of_covered_neurons),
                    str(total_neurons),
                    f"{coverage * 100:.2f}%",
                ]
            )

            self.pdf_writer.add_space(5)
            self.pdf_writer.add_table(data)

            self.pdf_writer.add_space(20)

        # Calculation of TKNC of the model

        data = [
            [
                "Coverage Metric",
                "TKNC Sum",
                "Number of Selected Neurons",
                "Mean of Top-K Neurons",
            ]
        ]

        self.pdf_writer.add_text(
            "Top-K Neuron Coverage (K = " + str(self.top_k_val) + ") Value of Model",
            font_size=16,
            is_bold=True,
        )
        self.pdf_writer.add_text(
            f"The table below shows Top-K Neuron Coverage value of the '"
            + self.model_info["name"]
            + "' model. Top-K Neuron Coverage (TKNC) is a metric used to evaluate the activation patterns and coverage of neurons in a deep neural network (DNN). It measures the percentage of neurons that are activated for a given set of input samples. The idea behind TKNC is to assess how well a set of input samples can activate different neurons in the network. How is it calculated? TKNC travels through all layers on a model one by one and ranks the neuron values of each layer in order from largest to smallest. Then it takes k neurons in each layer and adds it to a list. It then creates a value called 'TKNC Sum', which represents the sum of neurons in this list. The 'Number of Selected Neurons' value shows how many neurons were selected on the whole model as a result of k neurons from each layer. The 'Mean of Top-K Neurons' value shows the ratio of the 'TKNC Sum' value to the 'Number of Selected Neurons' value. The values in the table below, it was formed as a result of giving the '"
            + self.random_image_name
            + "' input in the data set to the model.",
            font_size=14,
            is_bold=False,
        )
        self.pdf_writer.add_space(5)

        (
            tknc_sum,
            num_of_selected_neurons,
            mean_top_k,
        ) = self.driver.get_tknc_coverage_of_model(self.sample, self.top_k_val)
        data.append(
            [
                "Top-K Neuron Coverage",
                f"{tknc_sum:.2f}",
                str(num_of_selected_neurons),
                f"{mean_top_k:.2f}",
            ]
        )

        self.pdf_writer.add_space(5)
        self.pdf_writer.add_table(data)

        self.pdf_writer.add_space(20)

        # Calculation of NBC of the model

        data = [
            [
                "Coverage Metric",
                "NBC Counter",
                "Number of Total Neurons",
                "Neuron Boundary Coverage",
            ]
        ]

        self.pdf_writer.add_text(
            "Neuron Boundary Coverage Value of Model (For "
            + str(len(self.samples))
            + " Inputs)",
            font_size=16,
            is_bold=True,
        )
        self.pdf_writer.add_text(
            f"The table below shows Neuron Boundary Coverage value of the '"
            + self.model_info["name"]
            + "' model. Neuron Boundary Coverage (NBC) is a metric used to evaluate the coverage of decision boundaries in a deep neural network (DNN). It measures the percentage of decision boundaries in the network that have been activated or crossed by the input samples. How is it calculated? NBC receives a random set of inputs from the user, and as a result of these inputs, it determines the maximum and minimum interval value for each layer. Then, for the input data to be checked, it is checked whether each neuron belonging to each layer is within the maximum and minimum range of this layer. If it is within this range, the 'NBC Counter' value is increased by one. The values in the table below, it was formed as a result of giving the '"
            + self.random_image_name
            + "' input in the data set to the model.",
            font_size=14,
            is_bold=False,
        )
        self.pdf_writer.add_space(5)

        nbc_counter, total_neurons, coverage = self.driver.get_nbc_coverage_of_model(
            self.samples, self.sample
        )
        data.append(
            [
                "Neuron Boundary Coverage",
                str(nbc_counter),
                str(total_neurons),
                f"{coverage * 100:.2f}%",
            ]
        )

        self.pdf_writer.add_space(5)
        self.pdf_writer.add_table(data)

        self.pdf_writer.add_space(20)

        # Calculation of MNC of the model

        data = [
            [
                "Threshold Intervals",
                "MNC Counter",
                "Number of Total Neurons",
                "Multisection Neuron Coverage",
            ]
        ]

        self.pdf_writer.add_text(
            "Multisection Neuron Coverage Value of Model",
            font_size=16,
            is_bold=True,
        )
        self.pdf_writer.add_text(
            f"The table below shows Multisection Neuron Coverage value of the '"
            + self.model_info["name"]
            + "' model. Multisection Neuron Coverage (MNC) specifically focuses on assessing the coverage of individual neurons within the model. The goal of MNC is to evaluate the degree to which the decisions made by individual neurons have been exercised by the test cases. It helps identify potential shortcomings in the model's behavior and reveal areas that may require further testing. It provides the user with the information of how many neurons are found according to the threshold value ranges given by the user. How is it calculated? The MNC receives threshold ranges from the user. Then, it evaluates all the neurons on the model and checks whether each neuron is within these threshold ranges. If the corresponding neuron is within this threshold value, it increases the 'MNC Counter' value found for the relevant range by one. The 'Multisection Neuron Coverage' value is the ratio of the 'MNC Counter' value to the number of all neurons on the model. The values in the table below, it was formed as a result of giving the '"
            + self.random_image_name
            + "' input in the data set to the model.",
            font_size=14,
            is_bold=False,
        )
        self.pdf_writer.add_space(5)

        counter_arr, total_neurons = self.driver.get_mnc_coverage_of_model(
            self.sample, self.node_intervals
        )

        for i, res in enumerate(counter_arr):
            data.append(
                [
                    f"{self.node_intervals[i][0]} - {self.node_intervals[i][1]}",
                    str(counter_arr[i]),
                    str(total_neurons),
                    f"{counter_arr[i] / total_neurons * 100:.2f}%",
                ]
            )

        self.pdf_writer.add_space(5)
        self.pdf_writer.add_table(data)

        self.pdf_writer.save()
