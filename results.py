import re
import csv
from io import StringIO
import locale


# Global list to store results
results = []
headers = ["Size & Degree"]


def process_csv_data(csv_reader, block_dimensions, first_block):
    values = [block_dimensions]
    for row_index, row in enumerate(csv_reader):
        if row_index != 0 and row:
            if first_block:
                headers.append(row[-3])
                values.append(locale.atof(row[-1]))
            else:
                values.append(locale.atof(row[-1]))

    values[1] /= (2 ** 30)
    values[2] /= 1000000000
    values[4] *= 2

    results.append(values)


def process_block(block_data, first_block):
    size = re.search(r'\d+', block_data[1]).group()
    degree = re.search(r'\d+', block_data[2]).group()

    block_dimensions = f"{size}x{degree}"


    csv_string = '\n'.join(block_data[4:])
    csv_reader = csv.reader(StringIO(csv_string))

    process_csv_data(csv_reader, block_dimensions, first_block)


def process_file(filename):
    start_pattern = re.compile(r"^==PROF== Connected to process \d+ .+$")

    with open(filename, 'r') as file:
        block_data = []
        inside_block = False
        first_block = True

        for line in file:
            if start_pattern.match(line):
                if inside_block:
                    # End of the current block, process it
                    process_block(block_data, first_block)
                    block_data = []
                    first_block = False

                # Start of a new block
                inside_block = True
            elif inside_block:
                block_data.append(line)

        # Process the last block if there was one
        if inside_block:
            process_block(block_data, first_block)


locale.setlocale(locale.LC_ALL, '')
process_file('./results/metrics.csv')


# Print the results in the desired format
print(headers)
for r in results:
    print(r)
