import os
from dataset.dataloader import SatDataset
import random
from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent

base = os.path.join(source_dir,  "../../data/int/applications")

uar_elevation_csv = base + "/elevation/outcomes_sampled_elevation_CONTUS_16_640_UAR_100000_0.csv"
uar_income_csv = base + "/income/outcomes_sampled_income_CONTUS_16_640_UAR_100000_0.csv"
uar_population_csv = base + "/population/outcomes_sampled_population_CONTUS_16_640_UAR_100000_0.csv"
uar_roads_csv = base + "/roads/outcomes_sampled_roads_CONTUS_16_640_UAR_100000_0.csv"
uar_treecover_csv = base + "/treecover/outcomes_sampled_treecover_CONTUS_16_640_UAR_100000_0.csv"
uar_nightlights_csv = base + "/nightlights/outcomes_sampled_nightlights_CONTUS_16_640_UAR_100000_0.csv"

elevation_col_title = "elevation"
income_col_title = "income"
population_col_title = "population"
roads_col_title = "length_S1400"
treecover_col_title = "treecover"
nightlights_col_title = "y"

class Task:
    def __init__(self, name, display_name, csv_file, col_name, code):
        self.name = name
        self.display_name = display_name
        self.csv_file = csv_file
        self.col_name = col_name
        self.code = code

elevation_task = Task("elevation", "Elevation", uar_elevation_csv, elevation_col_title, "El")
income_task = Task("income", "Income", uar_income_csv, income_col_title, "In")
population_task = Task("population", "Population", uar_population_csv, population_col_title, "Po")
roads_task = Task("roads", "Roads", uar_roads_csv, roads_col_title, "Rd")
treecover_task = Task("treecover", "Tree Cover", uar_treecover_csv, treecover_col_title, "Tr")
nightlights_task = Task("nightlights", "Nightlights", uar_nightlights_csv, nightlights_col_title, "Nl")

all_tasks = [elevation_task, income_task, population_task, roads_task, treecover_task, nightlights_task]
task_name_map = {task.name: task for task in all_tasks}
task_code_map = {task.code: task for task in all_tasks}

image_root = os.path.join(source_dir,  "../../data/raw/mosaiks_images") 

def create_dataset_all(transfrom):
    return SatDataset(all_tasks, image_root, transfrom)

def create_dataset_treecover(transfrom):
    return SatDataset([treecover_task], image_root, transfrom)

def create_dataset_income(transfrom):
    return SatDataset([income_task], image_root, transfrom)

# three random tasks chosen: elevation, roads, treecover
def create_dataset_three(transfrom):
    return SatDataset([treecover_task, elevation_task, roads_task], image_root, transfrom)

# four random tasks chosen: elevation, roads, income, nightlights
def create_dataset_four(transfrom):
    return SatDataset([elevation_task, roads_task, income_task, nightlights_task], image_root, transfrom)

def create_dataset_ablation(transfrom, hold_out_task_name, root_dir):
    tasks = all_tasks.copy()
    hold_out_task = task_name_map[hold_out_task_name]
    tasks.remove(hold_out_task)
    return SatDataset(tasks, image_root, transfrom)
