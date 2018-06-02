import os

DIR_NAME = 'race/UTKFace'

if __name__ == '__main__':
    for file_name in os.listdir(DIR_NAME):
        num_parts = len(file_name.split('_'))
        if num_parts != 4:
            print(f'Invalid file {DIR_NAME}/{file_name}')
