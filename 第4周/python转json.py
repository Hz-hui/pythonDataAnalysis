import json
import base64

def serialize_to_json(data):
    # print(type(data))
    new_data = json.dumps(data)
    # print(type(new_data))
    new_data = new_data.replace('True', 'true')
    new_data = new_data.replace('False', 'false')
    new_data = new_data.replace('None', 'null')

    if type(data) == list or type(data) == dict:
        new_data = new_data.replace("{", "{\n  ")
        new_data = new_data.replace(", ", ",\n  ")
        new_data = new_data.replace("}", "\n}")
        new_data = new_data.replace("[", "[\n  ")
        new_data = new_data.replace("]", "\n]")

    else:
        new_data = "  " + new_data

    return new_data

print(serialize_to_json(eval(input())))