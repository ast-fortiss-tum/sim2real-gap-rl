import json
import os

def process_value(key, value):
    """
    Process and print details about a value:
      - If it's a list, print the number of elements and a sample.
      - If it's a dictionary, print the number of keys.
      - If it's a string that represents a list (starts with '[' and ends with ']'),
        attempt to convert Python-style booleans to JSON booleans and parse it.
      - Otherwise, print its type.
    """
    if isinstance(value, list):
        print(f"Key: '{key}' contains a list with {len(value)} element(s).")
        #print(f"  Sample data: {value[:5]}{'...' if len(value) > 5 else ''}")
    elif isinstance(value, dict):
        print(f"Key: '{key}' contains a dictionary with {len(value)} key(s).")
    elif isinstance(value, str) and value.strip().startswith('[') and value.strip().endswith(']'):
        # Preprocess the string to convert Python booleans to JSON booleans.
        value_to_parse = value.replace("True", "true").replace("False", "false")
        try:
            parsed_value = json.loads(value_to_parse)
            if isinstance(parsed_value, list):
                print(f"Key: '{key}' contains a string that represents a list with {len(parsed_value)} element(s).")
                #print(f"  Sample data: {parsed_value[:5]}{'...' if len(parsed_value) > 5 else ''}")
            else:
                print(f"Key: '{key}' is a string that parsed into a {type(parsed_value).__name__}.")
        except json.JSONDecodeError as e:
            print(f"Key: '{key}' is a string that could not be parsed as JSON. Error: {e}")
    else:
        print(f"Key: '{key}' is of type '{type(value).__name__}' (no nested elements to count).")

def print_json_keys_and_elements(file_path):
    """
    Load a JSON file, convert it to a dictionary, and for each key:
      - Print the key.
      - For lists or strings representing lists, print the number of elements.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if isinstance(data, dict):
            print(f"Processing file: '{file_path}'")
            for key, value in data.items():
                process_value(key, value)
        else:
            print(f"The file '{file_path}' does not contain a JSON object at the top level.")
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")

if __name__ == '__main__':
    # Option 1: Process a single file
    #json_file = '/home/cubos98/catkin_ws/src/Vehicle/results_reality'  # Replace with your JSON file path
    #print_json_keys_and_elements(json_file)
    
    # Option 2: Process all JSON files in a directory
    # Uncomment the following block if you want to process multiple files in a folder.
    
    directory = '/home/cubos98/catkin_ws/src/Vehicle/results_reality'  # Replace with the directory containing your JSON files
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            print_json_keys_and_elements(file_path)
            print('-' * 40)

    
    

    

    


