import os
import ast

def generate_docstring(node):
    """Generate a basic docstring for a class or function."""
    if isinstance(node, ast.ClassDef):
        return f'"""Class {node.name}."""\n'
    elif isinstance(node, ast.FunctionDef):
        params = ', '.join([f'{arg.arg}: {arg.annotation.id if arg.annotation else "Any"}' for arg in node.args.args])
        return f'"""Function {node.name}.\n\nArgs:\n    {params}\n\nReturns:\n    None."""\n'
    return ''

def add_docstring_to_file(file_path):
    with open(file_path, 'r') as file:
        source = file.read()

    tree = ast.parse(source)
    new_source = source.splitlines(keepends=True)

    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            docstring = generate_docstring(node)
            if not ast.get_docstring(node):
                # Insert docstring right after the definition
                line_number = node.lineno - 1
                new_source.insert(line_number + 1, docstring)

    with open(file_path, 'w') as file:
        file.writelines(new_source)

def generate_docstrings(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                add_docstring_to_file(os.path.join(root, file))

# Call the function with your library path
generate_docstrings('Optimizer/optimiz_it')