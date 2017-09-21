import json


from ..utils.general import init_dir


def load_config_json(path_to_json):
    with open(path_to_json) as f:
        data = json.load(f)
        fill_values(data)
        merge_lists(data)
        return data


def init_directories(data):
    """If an entry starts with dir, initializes the repo"""
    if type(data) is list:
        for value in data:
            init_directories(value)
    elif type(data) is dict:
        for name, value in data.items():
            if type(value) is unicode and name[:3] == "dir":
                init_dir(value)
            else:
                init_directories(value)


def merge_lists(data):
    """If data is a list of strings, concatenate this entry

    Ex:
        data = {"x": ["hello", "world"]}
        -> {"x": "helloworld"}
    Args:
        data: (dict or list or string or int or float)

    Returns:
        data

    """
    if type(data) is list:
        if type(data[0]) is unicode:
            return "".join(data)
        else:
            for i in range(len(data)):
                data[i] = merge_lists(data[i])
    elif type(data) is dict:
        for name, value in data.items():
            data[name] = merge_lists(value)
    return data



def fill_values(data, values=None):
    """If data value starts with _ -> replace with value from values

    Ex:
        data = {"x": 5, "y": {"z": "_x"}}
        -> ("x": 5, "y": {"z": 5})
    Args:
        data: (dict or list or string or int or float...)
        values: (dict)

    Returns:
        data
    """
    if values is None:
        values = data if type(data) is dict else {}

    if type(data) is unicode:
        if data[0] == "_":
            val_name = data[1:]
            if val_name in values:
                data = values[val_name]
            else:
                print("{} not found".format(val_name))
    elif type(data) is dict:
        for name, value in data.items():
            data[name] = fill_values(value, values)
    elif type(data) is list:
        for i in range(len(data)):
            data[i] = fill_values(data[i], values)

    return data
