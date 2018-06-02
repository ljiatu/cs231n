import os

if __name__ == '__main__':
    for subdir in os.listdir('imdb_wiki'):
        for file_name in os.listdir(f'imdb_wiki/{subdir}'):
            parts = file_name.split('.')[0].split('_')
            dob = int(parts[-2].split('-')[0])
            year = int(parts[-1])
            age = year - dob
            if age < 0 or age > 100:
                file_path = f'imdb_wiki/{subdir}/{file_name}'
                print(f'Removing {file_path}')
                os.remove(f'imdb_wiki/{subdir}/{file_name}')
