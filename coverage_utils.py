class CoverageUtils:
    @staticmethod
    def is_there_sign_change(
        k, i, model_architecture_dict_for_tI, model_architecture_dict_for_tII
    ):
        neuron_value_for_tI = model_architecture_dict_for_tI[str(k)][
            "after_act_func_values"
        ][0][i]
        neuron_value_for_tII = model_architecture_dict_for_tII[str(k)][
            "after_act_func_values"
        ][0][i]

        if neuron_value_for_tI * neuron_value_for_tII < 0:
            return True
        else:
            return False

    @staticmethod
    def is_there_sign_sign_change(
        k, i, j, model_architecture_dict_for_tI, model_architecture_dict_for_tII
    ):
        flag = True

        for neuron_index in range(
            len(model_architecture_dict_for_tI[str(k)]["after_act_func_values"][0])
        ):
            if i == neuron_index:
                if not CoverageUtils.is_there_sign_change(
                    k,
                    i,
                    model_architecture_dict_for_tI,
                    model_architecture_dict_for_tII,
                ):
                    flag = False
                    break
            else:
                if CoverageUtils.is_there_sign_change(
                    k,
                    neuron_index,
                    model_architecture_dict_for_tI,
                    model_architecture_dict_for_tII,
                ):
                    flag = False
                    break

        if not CoverageUtils.is_there_sign_change(
            k + 1, j, model_architecture_dict_for_tI, model_architecture_dict_for_tII
        ):
            flag = False

        return flag
