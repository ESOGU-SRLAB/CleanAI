class ModelArchitectureUtils:
    @staticmethod
    def get_before_values_for_specific_layer(activation_info, layer_index):
        return activation_info[str(layer_index)]["before_act_func_values"]

    @staticmethod
    def get_before_values_for_all_layers(activation_info):
        before_values_for_all_layers = []
        for layer_index in range(len(activation_info)):
            before_values = ModelArchitectureUtils.get_before_values_for_specific_layer(
                activation_info, layer_index
            )
            before_values_for_all_layers.append(before_values)

        return before_values_for_all_layers

    @staticmethod
    def get_after_values_for_specific_layer(activation_info, layer_index):
        return activation_info[str(layer_index)]["after_act_func_values"]

    @staticmethod
    def get_after_values_for_all_layers(activation_info):
        after_values_for_all_layers = []
        for layer_index in range(len(activation_info)):
            after_values = ModelArchitectureUtils.get_after_values_for_specific_layer(
                activation_info, layer_index
            )
            after_values_for_all_layers.append(after_values)

        return after_values_for_all_layers

    @staticmethod
    def get_after_values_for_multiple_inputs(activation_infos_of_inputs):
        after_values_for_multiple_inputs = []
        for activation_info_of_input in activation_infos_of_inputs:
            after_values_for_all_layers = (
                ModelArchitectureUtils.get_after_values_for_all_layers(
                    activation_info_of_input
                )
            )
            after_values_for_multiple_inputs.append(after_values_for_all_layers)

        return after_values_for_multiple_inputs

    @staticmethod
    def get_neuron_value_from_after_values(activation_info, layer_index, neuron_index):
        neuron_value = ModelArchitectureUtils.get_after_values_for_specific_layer(
            activation_info, layer_index
        )[neuron_index]
        return neuron_value
