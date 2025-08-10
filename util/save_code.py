import inspect
import json
import os
import shutil
import sys


def save_dependencies_files(root_path,args=None):
    """
    Save dependencies of the running script
    :param root_path:
    :param args:
    :return:
    """
    abs_cwd = os.path.abspath(os.getcwd())
    # 获取当前加载的所有模块
    modules = list(sys.modules.values())
    # 获取所有依赖的文件的路径
    dependency_files = []
    for module in modules:
        try:

            file_path = os.path.abspath(inspect.getfile(module))
            if os.path.isfile(file_path) and os.path.commonpath([abs_cwd, file_path]) == abs_cwd:
                dependency_files.append(file_path)
        except TypeError:
            # 忽略无法获取文件路径的模块
            continue
    dependency_files = list(set(dependency_files))
    for file_path in dependency_files:
        save_path = os.path.join(root_path, os.path.relpath(file_path, abs_cwd))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)
        shutil.copy(file_path, save_path)

    if args is not None:
        with open(os.path.join(root_path,'args.json'),'w',encoding='utf-8') as file:
            json.dump(vars(args), file, indent=4)
    return dependency_files
