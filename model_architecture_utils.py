class ModelArchitectureUtils:
    @staticmethod
    def get_before_values_for_specific_layer(model_architecture_dict, layer_index):
        return model_architecture_dict[str(layer_index)]["before_act_func_values"]

    @staticmethod
    def get_before_values_for_all_layers(model_architecture_dict):
        before_values_for_all_layers = []
        for layer_index in range(len(model_architecture_dict)):
            before_values = ModelArchitectureUtils.get_before_values_for_specific_layer(
                model_architecture_dict, layer_index
            )
            before_values_for_all_layers.append(before_values)

        return before_values_for_all_layers

    @staticmethod
    def get_after_values_for_specific_layer(model_architecture_dict, layer_index):
        return model_architecture_dict[str(layer_index)]["after_act_func_values"]

    @staticmethod
    def get_after_values_for_all_layers(model_architecture_dict):
        after_values_for_all_layers = []
        for layer_index in range(len(model_architecture_dict)):
            after_values = ModelArchitectureUtils.get_after_values_for_specific_layer(
                model_architecture_dict, layer_index
            )
            after_values_for_all_layers.append(after_values)

        return after_values_for_all_layers

    @staticmethod
    def get_after_values_for_multiple_inputs(model_architecture_dicts_of_inputs):
        after_values_for_multiple_inputs = []
        for model_architecture_dict_of_input in model_architecture_dicts_of_inputs:
            after_values_for_all_layers = (
                ModelArchitectureUtils.get_after_values_for_all_layers(
                    model_architecture_dict_of_input
                )
            )
            after_values_for_multiple_inputs.append(after_values_for_all_layers)

        return after_values_for_multiple_inputs
