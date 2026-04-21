import os

def print_tree(root='.', ignore=('__pycache__', '.vscode', '.venv', '.DS_Store', '.git')):
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in sorted(dirs) if d not in ignore]
        level = dirpath.replace(root, '').count(os.sep)
        indent = '│   ' * level + '├── '
        print(f'{indent}{os.path.basename(dirpath)}/')
        subindent = '│   ' * (level + 1) + '├── '
        for file in sorted(files):
            print(f'{subindent}{file}')

print_tree()