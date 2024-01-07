import ast
import os


def find_imports_in_init(init_path):
    imported_funcs_classes = []

    with open(init_path, "r") as f:
        tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_funcs_classes.append(alias.name.split(".")[-1])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_funcs_classes.append(alias.name)

    return imported_funcs_classes


def find_all_funcs_in_folder(folder_path, init_path):
    funcs_classes = []
    imported_funcs_classes = find_imports_in_init(init_path)
    not_imported = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r") as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) or isinstance(
                            node, ast.ClassDef
                        ):
                            name = node.name
                            funcs_classes.append(
                                f"{root}/{file}: {type(node).__name__} {name}"
                            )
                            if name not in imported_funcs_classes:
                                not_imported.append(
                                    f"{root}/{file}:"
                                    f" {type(node).__name__} {name}"
                                )

    return funcs_classes, not_imported


funcs_classes, not_imported = find_all_funcs_in_folder(
    "zeta/nn/modules", "zeta/nn/modules/__init__.py"
)
print("All functions and classes:")
print(funcs_classes)
print("Not imported in __init__.py:")
print(not_imported)


def write_to_file(file_path, list):
    with open(file_path, "w") as f:
        for item in list:
            f.write(f"{item}\n")


write_to_file("all_funcs_classes.txt", funcs_classes)
write_to_file("not_imported.txt", not_imported)
