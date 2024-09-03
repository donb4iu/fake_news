import json

def read_and_convert_to_json():
    """
    Prompts for a file name, reads the text file, escapes all double quotes,
    and converts the contents to JSON format with a single key-value pair.
    """
    # Prompt the user for the file name
    file_name = input("Enter the name of the text file (with extension): ")

    try:
        # Open and read the contents of the file
        with open(file_name, 'r') as file:
            content = file.read()

        # Escape all double quotes in the content
        escaped_content = content.replace('"', '\\"')

        # Create a dictionary with the key "text" and the escaped content as the value
        json_data = {
            "text": escaped_content
        }

        # Convert the dictionary to a JSON string
        json_string = json.dumps(json_data, indent=4)

        # Define the output file name
        output_file_name = file_name.split('.')[0] + "_output.json"

        # Write the JSON string to the output file
        with open(output_file_name, 'w') as output_file:
            output_file.write(json_string)

        print(f"JSON file '{output_file_name}' created successfully.")

    except FileNotFoundError:
        print(f"The file '{file_name}' does not exist. Please try again.")

if __name__ == "__main__":
    read_and_convert_to_json()
