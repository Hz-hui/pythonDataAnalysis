import pickle
import base64

def get_sorted_key_value(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        a = sorted(data.keys())
        b = [data[key] for key in a]

        c = [a, b]
    return c
