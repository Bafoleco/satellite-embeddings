import tasks
import numpy as np

def mosaiks_format_to_map(X, ids_X, embeddings, dim=2048):
    """
    Convert the mosaiks format to a map
    """
    mosaiks_map = {}
    for i in range(len(ids_X)):
        if ids_X[i] in embeddings:
            mosaiks_map[ids_X[i]] = X[i][:dim]
    return mosaiks_map
 
def parse_tasks(model_name):
    """
    Parse the tasks from the model name
    """
    task_str = model_name.split("_")[2]

    print("Task str: ", task_str)

    # split every two characters into a list
    task_codes = [task_str[i:i+2] for i in range(0, len(task_str), 2)]

    print("Tasks: ", task_codes)

    task_list = [tasks.task_code_map[task_code] for task_code in task_codes]

    return task_list

def convert_map_to_nparray(embedding_map):
    """
    Convert the embedding map to a numpy array
    """
    X = []
    ids_X = []
    for key in embedding_map:
        X.append(embedding_map[key])
        ids_X.append(key)
    return np.array(X), np.array(ids_X)