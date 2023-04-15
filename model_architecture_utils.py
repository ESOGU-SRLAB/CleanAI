class ModelArchitectureUtils:
    @staticmethod
    def get_after_values_for_specific_layer(model_architecture_dict, layer_index):
        return model_architecture_dict[str(layer_index)]['after_act_func_values']
    
    @staticmethod
    def get_after_values_for_all_layers(model_architecture_dict):
        after_values_for_all_layers = []
        for layer_index in range(len(model_architecture_dict)):
            after_values = ModelArchitectureUtils.get_after_values_for_specific_layer(model_architecture_dict, layer_index)
            after_values_for_all_layers.append(after_values)

        return after_values_for_all_layers

