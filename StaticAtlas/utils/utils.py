import os
import shutil

def create_new_versioned_directory(base_name='./version', start_version=0):
    # Check if version_0 exists
    version = start_version
    while os.path.exists(f'{base_name}_{version}'):
        version += 1
    new_version = f'{base_name}_{version}'
    os.makedirs(new_version)
    print(f'Created save repository: {new_version}')
    return new_version



if __name__ == "__main__":
    create_new_versioned_directory("./save/version")