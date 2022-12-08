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
roads_col_title = "length"
treecover_col_title = "treecover"
nightlights_col_title = "y"

class Task:
    def __init__(self, name, display_name, csv_file, col_name, code, log=False):
        self.name = name
        self.display_name = display_name
        self.csv_file = csv_file
        self.col_name = col_name
        self.code = code
        self.log = log

elevation_task = Task("elevation", "Elevation", uar_elevation_csv, elevation_col_title, "El")
income_task = Task("income", "Income", uar_income_csv, income_col_title, "In")
population_task = Task("population", "Population", uar_population_csv, population_col_title, "Po", log=True)
roads_task = Task("roads", "Roads", uar_roads_csv, roads_col_title, "Rd")
treecover_task = Task("treecover", "Tree Cover", uar_treecover_csv, treecover_col_title, "Tr")
nightlights_task = Task("nightlights", "Nightlights", uar_nightlights_csv, nightlights_col_title, "Nl", log=True)

all_tasks = [elevation_task, income_task, population_task, roads_task, treecover_task, nightlights_task]

image_root = os.path.join(source_dir,  "../../data/raw/mosaiks_images") 

def create_dataset_all(transfrom):
    return SatDataset(all_tasks, image_root, transfrom)

def create_dataset_treecover(transfrom, image_root=image_root):
    return SatDataset([treecover_task], image_root, transfrom)

def create_dataset_income(transfrom):
    return SatDataset([income_task], image_root, transfrom)

def create_dataset_ablation(transfrom, hold_out_task_name = None):
    tasks = [elevation_task, roads_task, income_task, treecover_task]

    if (hold_out_task_name):
        hold_out_task = task_name_map[hold_out_task_name]
        tasks.remove(hold_out_task)

    return SatDataset(tasks, image_root, transfrom)

# acs task
B08303_csv = os.path.join(base, "ACS", "B08303", "outcomes_sampled_B08303_CONTUS_16_640_UAR_100000_0.csv")
B15003_csv = os.path.join(base, "ACS", "B15003", "outcomes_sampled_B15003_CONTUS_16_640_UAR_100000_0.csv")
B19013_csv = os.path.join(base, "ACS", "B19013", "outcomes_sampled_B19013_CONTUS_16_640_UAR_100000_0.csv")
B19301_csv = os.path.join(base, "ACS", "B19301", "outcomes_sampled_B19301_CONTUS_16_640_UAR_100000_0.csv")
B22010_csv = os.path.join(base, "ACS", "B22010", "outcomes_sampled_B22010_CONTUS_16_640_UAR_100000_0.csv")
B25001_csv = os.path.join(base, "ACS", "B25001", "outcomes_sampled_B25001_CONTUS_16_640_UAR_100000_0.csv")
B25002_csv = os.path.join(base, "ACS", "B25002", "outcomes_sampled_B25002_CONTUS_16_640_UAR_100000_0.csv")
B25017_csv = os.path.join(base, "ACS", "B25017", "outcomes_sampled_B25017_CONTUS_16_640_UAR_100000_0.csv")
B25035_csv = os.path.join(base, "ACS", "B25035", "outcomes_sampled_B25035_CONTUS_16_640_UAR_100000_0.csv")
B25071_csv = os.path.join(base, "ACS", "B25071", "outcomes_sampled_B25071_CONTUS_16_640_UAR_100000_0.csv")
B25077_csv = os.path.join(base, "ACS", "B25077", "outcomes_sampled_B25077_CONTUS_16_640_UAR_100000_0.csv")
C17002_csv = os.path.join(base, "ACS", "C17002", "outcomes_sampled_C17002_CONTUS_16_640_UAR_100000_0.csv")

B08303_task = Task("ttt_work", "Travel time to work", B08303_csv, "Val", "Tw")
B15003_task = Task("pct_bach", "Percent Bachelor’s Degree", B15003_csv, "Val", "Pb")
B19013_task = Task("md_house_income", "Median Household Income", B19013_csv, "Val", "Mh")
B19301_task = Task("per_capita_income", "Per Capita Income", B19301_csv, "Val", "Pi")
B22010_task = Task("pct_snap", "Percent food stamp/snap", B22010_csv, "Val", "Ps")
B25001_task = Task("num_housing_units", "Number of Housing Units", B25001_csv, "Val", "Nh")
B25002_task = Task("pct_vacant", "Percent Vacant", B25002_csv, "Val", "Pv")
B25017_task = Task("num_rooms", "Number of Rooms", B25017_csv, "Val", "Nr")
B25035_task = Task("structure_age", "Structure Age", B25035_csv, "Val", "Sa")
B25071_task = Task("median_income", "Median Income", B25071_csv, "Val", "Mi")  
B25077_task = Task("median_value", "Median House Value", B25077_csv, "Val", "Mv")
C17002_task = Task("pct_poverty", "Percentage Below Poverty Level", C17002_csv, "Val", "Pp")


acs_tasks = [B08303_task, B15003_task, B19013_task, B19301_task, B22010_task, B25001_task, B25002_task, B25017_task, B25035_task, B25071_task, B25077_task, C17002_task]

task_name_map = {task.name: task for task in acs_tasks + all_tasks}
task_code_map = {task.code: task for task in acs_tasks + all_tasks}
