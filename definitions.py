def MODEL_INFORMATIONS_TITLE():
    return "Model Informations"


def MODEL_INFORMATIONS_DESCRIPTION(model_name):
    return (
        "The table below shows general information about the '"
        + model_name
        + "' model."
    )


def MODEL_INFORMATIONS_ROW_HEADERS(model_name):
    return [model_name]


def MODEL_INFORMATIONS_COL_HEADERS():
    return ["Model name", "Total params", "Number of layers"]


def MODEL_INFORMATIONS_TABLE_DATA(model_name, total_params, num_layers):
    return [
        model_name,
        str(total_params),
        str(num_layers),
    ]


def MODEL_COVERAGE_TITLE():
    return "Coverage Values of Layers"


def MODEL_COVERAGE_DESCRIPTION(model_name):
    return (
        "The table below shows coverage values about the '"
        + model_name
        + "' model's all layers."
    )


def MODEL_COVERAGE_ROW_HEADERS():
    return ["Layer index"]


def MODEL_COVERAGE_COL_HEADERS():
    return [
        "Layer index",
        "Number of covered neurons",
        "Number of total neurons",
        "Coverage value",
    ]


def MODEL_COVERAGE_TABLE_ONE_ROW(
    layer_index, num_of_covered_neurons, total_neurons, coverage
):
    return (
        [
            str(layer_index),
            str(num_of_covered_neurons),
            str(total_neurons),
            str(coverage),
        ],
    )
