import os
import json
import numpy as np


def saving(weights: list[np.ndarray], file_name: str) -> None:
    """
    saves the weights in the savedweights folder
    If savedweigths doesn't exist yet, it will create the folder.
    """
    if not os.path.exists("./savedweights"):
        os.makedirs("./savedweights")

    with open(f"./savedweights/{file_name}.json", "w") as file:
        json.dump({index: weight.tolist() for index, weight in
                   enumerate(weights)},
                  file, indent=4)


def loading(file_name: str) -> list[np.ndarray]:
    """loads the weights from the savedweigths directory"""
    if not os.path.exists("./savedweights/"):
        raise FileNotFoundError("savedweights directory not found")
    elif not os.path.exists(f"./savedweights/{file_name}.json"):
        raise FileNotFoundError(f"{file_name} file not found")

    try:
        with open(f"./savedweights/{file_name}.json", "r") as file:
            dict_of_file = json.load(file)

        return_list = []
        for key in dict_of_file.keys():
            return_list.append(np.array(dict_of_file[key]))

        return return_list
    except ValueError:
        raise ValueError("couldn't import from quicksave,"
                         + " the file might be corrupted")
