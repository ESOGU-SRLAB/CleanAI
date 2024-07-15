"""
Overview: The purpose of this class is to prepare functions that return 
activation values in a particular layer or activation value of a particular 
neuron of a particular layer.

Maintainers: - Osman Çağlar - cglrr.osman@gmail.com
             - Abdul Hannan Ayubi - abdulhannanayubi38@gmail.com
"""


class ModelArchitectureUtils:
    @staticmethod
    def get_before_values_for_specific_layer(activation_info, layer_index):
        """
        This function is used to return the 'before activation values' of the neurons in
        the specific layer with index.

        Args:
            activation_info (dict): Dictionary that contains the activation values of the
            neurons in the layers.
        Returns:
            before_values (list): List of the 'before activation values' of the neurons in
            the specific layer with index.
        """
        return activation_info[str(layer_index)]["before_act_func_values"]

    @staticmethod
    def get_before_values_for_all_layers(activation_info):
        """
        This function is used to return the 'before activation values' of the neurons in
        all layers.

        Args:
            activation_info (dict): Dictionary that contains the activation values of the
        Returns:
            before_values_for_all_layers (list): List of the 'before activation values' of
            the neurons in all layers.
        """
        before_values_for_all_layers = []
        for layer_index in range(len(activation_info)):
            before_values = ModelArchitectureUtils.get_before_values_for_specific_layer(
                activation_info, layer_index
            )
            before_values_for_all_layers.append(before_values)

        return before_values_for_all_layers

    @staticmethod
    def get_after_values_for_specific_layer(activation_info, layer_index):
        """
        This function is used to return the 'after activation values' of the neurons in
        the specific layer with index.

        Args:
            activation_info (dict): Dictionary that contains the activation values of the
            neurons in the layers.
        Returns:
            after_values (list): List of the 'after activation values' of the neurons in
            the specific layer with index.
        """
        return activation_info[str(layer_index)]["after_act_func_values"]

    @staticmethod
    def get_after_values_for_all_layers(activation_info):
        """
        This function is used to return the 'after activation values' of the neurons in
        all layers.

        Args:
            activation_info (dict): Dictionary that contains the activation values of the
        Returns:
            after_values_for_all_layers (list): List of the 'after activation values' of
            the neurons in all layers.
        """
        after_values_for_all_layers = []
        for layer_index in range(len(activation_info)):
            after_values = ModelArchitectureUtils.get_after_values_for_specific_layer(
                activation_info, layer_index
            )
            after_values_for_all_layers.append(after_values)

        return after_values_for_all_layers

    @staticmethod
    def get_after_values_for_multiple_inputs(activation_infos_of_inputs):
        """
        This function is used to return the 'after activation values' of the neurons in
        all layers for multiple inputs.

        Args:
            activation_infos_of_inputs (list): List of dictionaries that contains the
            activation values of the neurons in the layers for multiple inputs.
        Returns:
            after_values_for_multiple_inputs (list): List of the 'after activation values'
            of the neurons in all layers for multiple inputs.
        """
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
        """
        This function is used to return the 'after activation values' of the neuron in
        the specific layer and specific neuron with index.

        Args:
            activation_info (dict): Dictionary that contains the activation values of the
            neurons in the layers.
            layer_index (int): Index of the layer.
            neuron_index (int): Index of the neuron.
        Returns:
            neuron_value (float): 'After activation value' of the neuron in the specific
            layer and specific neuron with index.
        """
        neuron_value = ModelArchitectureUtils.get_after_values_for_specific_layer(
            activation_info, layer_index
        )[neuron_index]
        return neuron_value
