import os

def merge_files():
    base_dir = r"C:\Users\w\Desktop\A4\raw_data"
    files_to_merge = ["cuu25pbu.txt", "Person_1.txt", "Person_2.txt"]
    output_file = os.path.join(base_dir, "both.txt")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in files_to_merge:
            file_path = os.path.join(base_dir, fname)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")

if __name__ == "__main__":
    merge_files()
