import numpy as np
import os

def generate_inverted_file(input_file: str) -> None:
    output_file = f"data/inverted/{input_file.split('/')[-1]}"

    with open(output_file, "a") as output:
        with open(input_file, "r") as input:
            for line in input:
                inverted_line = line.replace("-1","0").replace("1","-1").replace("0","1")
                output.write(inverted_line)

file_list = os.popen('find data/opposite-of-inverted/ -type f').read().splitlines()

for file in file_list:
    generate_inverted_file(file)