import os
import json
import numpy as np


def save_to_json(item, file_path: str, file_name: str) -> None:
    """
    saves lists to json files
    """
    file_name = file_name.removesuffix(".json")

    if not os.path.exists("./" + file_path):
        os.makedirs("./" + file_path)

    with open(f"./{file_path}/{file_name}.json", "w+") as file:
        json.dump(item, file, indent=4)

def saving_weights_to_json(weights: list[np.ndarray], file_name: str) -> None:
    """
    saves the weights to the 'savedweights' folder as a .json file
    If 'savedweigths' doesn't exist yet, it will create the folder.
    """

    save_to_json({index: weight.tolist() for index, weight in
                   enumerate(weights)},
                   "savedweights", file_name)

def loading_weights_from_json(file_name: str) -> list[np.ndarray]:
    """loads the weights from the 'savedweights' directory"""
    
    try:
        dict_of_file = loading_from_json("savedweights", file_name)
        return_list = []
        for key in dict_of_file.keys():
            return_list.append(np.array(dict_of_file[key]))

        return return_list
    except ValueError:
        raise ValueError(f"couldn't import weights from {file_name}.json,"
                         + " the file might be corrupted")

def loading_from_json(file_path: str, file_name: str) -> list[np.ndarray]:
    """loads the weights from the 'savedweights' directory"""
    file_name = file_name.removesuffix(".json")

    if not os.path.exists("./" + file_path):
        raise FileNotFoundError(f"{file_path} directory not found")
    elif not os.path.exists(f"./{file_path}/{file_name}.json"):
        raise FileNotFoundError(f"{file_name} file not found")

    try:
        with open(f"./{file_path}/{file_name}.json", "r+") as file:
            return json.load(file)
    except ValueError:
        raise ValueError(f"couldn't import from {file_name}.json,"
                         + " the file might be corrupted")
