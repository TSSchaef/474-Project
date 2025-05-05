def load_label_mapping(mapping_file):
    label_to_char = {}
    try:
        with open(mapping_file, 'r') as file:
            for line in file:
                label, char_code = line.split()
                label_to_char[int(label)] = chr(int(char_code))
    except Exception as e:
        print(f"Error loading label mapping: {e}")
    return label_to_char

# Example usage
mapping_file = "../data/archive/emnist-mnist-mapping.txt"
label_mapping = load_label_mapping(mapping_file)
print(label_mapping)  # Prints the label-to-character mapping
