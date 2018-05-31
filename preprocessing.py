import os


for subdir in os.listdir('imdb_crop'):
    for file_name in os.listdir(f'imdb_crop/{subdir}'):
        parts = file_name.split('.')[0].split('_')
        dob = int(parts[2].split('-')[0])
        year = int(parts[3])
        age = year - dob
        if age < 0 or age > 100:
            print(f'imdb_crop/{subdir}/file_name')
