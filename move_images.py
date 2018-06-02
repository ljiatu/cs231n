import os


for subdir in os.listdir('wiki_norm'):
    for file_name in os.listdir(f'wiki_norm/{subdir}'):
        file_path = f'wiki_norm/{subdir}/{file_name}'
        os.rename(file_path, f'imdb_wiki/{subdir}/{file_name}')
