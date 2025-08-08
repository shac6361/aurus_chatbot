import ollama
import os  # import os module

directory = 'aurus\input'  # set directory path
model_name = 'minicpm-v'

# Model options: -
# gemma3
# llava
# minicpm-v
# mistral-small3.1 model requires more system memory


for entry in os.scandir(directory):  
    if entry.is_file():  # check if it's a file
        print(entry.path)
        response = ollama.generate(model=model_name, prompt="What's in this image?  "+entry.path)
        print(response['response'])

        with open("aurus\output\olama_output_"+model_name+".txt", "w") as file_object:
            file_object.write(response['response'])

        # Appending to a file
        # with open("output.txt", "a") as file_object:
        #     file_object.write("\nThis line is appended.")