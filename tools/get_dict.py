import json

dict_path = ''
out_path = ''

def get_dict(dict_path, out_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    keys = [key for key in data]
    with open(out_path, 'w', encoding='utf-8') as output_file:
        json.dump(keys, output_file, indent=2, ensure_ascii=False)

get_dict(dict_path, out_path)