import os
import re

def sort_files(filename):
    match = re.match(r"^a(\d+)_", filename)
    if match:
        return int(match.group(1))
    return float('inf')

def sort_dirs(dirname):
    match = re.search(r"(\d+)$", dirname)
    if match:
        return int(match.group(1))
    return dirname

def main():
    base_dir = os.path.join('raw_data', 'cuu25pbu')
    output_file = 'cuu25pbu.txt'

    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    subdirs.sort(key=sort_dirs)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for subdir in subdirs:
            subdir_path = os.path.join(base_dir, subdir)
            files = [f for f in os.listdir(subdir_path) if f.endswith('.txt')]

            # Sort files by a1, a2, a3, etc.
            files.sort(key=sort_files)

            for file_name in files:
                file_path = os.path.join(subdir_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as in_f:
                    content = in_f.read()
                    out_f.write(content)
                    # ensure newline between files just in case
                    if content and not content.endswith('\n'):
                        out_f.write('\n')

if __name__ == '__main__':
    main()

