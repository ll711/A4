import numpy as np
import pandas as pd
import os

def convert_npy_to_excel(folder_path):
    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} does not exist.")
        return

    # Find all .npy files
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    for file_name in npy_files:
        npy_path = os.path.join(folder_path, file_name)
        excel_path = os.path.join(folder_path, file_name.replace('.npy', '.xlsx'))

        # Load numpy data
        data = np.load(npy_path, allow_pickle=True)

        # Convert to DataFrame
        try:
            if data.ndim == 1:
                df = pd.DataFrame(data)
            elif data.ndim == 2:
                df = pd.DataFrame(data)
            else:
                # Try flattening for higher dimensional data
                df = pd.DataFrame(data.reshape(data.shape[0], -1))
            
            # Save to Excel without index and header
            df.to_excel(excel_path, index=False, header=False)
            print(f"Successfully converted {file_name} to {os.path.basename(excel_path)}")
        except Exception as e:
            print(f"Error occurred while converting {file_name}: {e}")

if __name__ == '__main__':
    convert_npy_to_excel('process_data')
