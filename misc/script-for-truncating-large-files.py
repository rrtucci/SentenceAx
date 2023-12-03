"""

Use this script to produce a file that is a short excerpt of a huge one.

"""


def delete_lines_from_file(fp, start_line):
    try:
        # Create a temporary file to store the lines to keep
        temp_fp = fp + ".temp"

        # Open the input file for reading and the temp file for writing
        with open(fp, "r", encoding="utf-8") as f, \
            open(temp_fp, "w", encoding="utf-8") as temp_f:
            line_number = 1
            for line in f:
                # If the current line number is less than the start_line, keep the line
                if line_number < start_line:
                    temp_f.write(line)
                line_number += 1

        # Replace the original file with the temp file
        import os
        os.replace(temp_fp, fp)

        print(f"Deleted lines from {start_line} onwards in {fp}")

    except FileNotFoundError:
        print(f"File '{fp}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Specify the file path and line number to start deleting from
fp = 'small_cctags.txt'
start_line = 1001

delete_lines_from_file(fp, start_line)
