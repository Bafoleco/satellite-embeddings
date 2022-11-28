from dataloader import SatDataset

base = "../data/int/applications"
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
    def __init__(self, name, display_name, csv_file, col_name):
        self.name = name
        self.display_name = display_name
        self.csv_file = csv_file
        self.col_name = col_name

elevation_task = Task("elevation", "Elevation", uar_elevation_csv, elevation_col_title)
income_task = Task("income", "Income", uar_income_csv, income_col_title)
population_task = Task("population", "Population", uar_population_csv, population_col_title)
roads_task = Task("roads", "Roads", uar_roads_csv, roads_col_title)
treecover_task = Task("treecover", "Tree Cover", uar_treecover_csv, treecover_col_title)
nightlights_task = Task("nightlights", "Nightlights", uar_nightlights_csv, nightlights_col_title)

all_tasks = [elevation_task, income_task, population_task, roads_task, treecover_task, nightlights_task]
task_map = {task.name: task for task in all_tasks}

image_root = "../data/raw/mosaiks_images"

def create_dataset_all(transfrom):
    return SatDataset(all_tasks, image_root, transfrom)

def create_dataset_treecover(transfrom, image_root="../data/raw/mosaiks_images"):
    return SatDataset([treecover_task], image_root, transfrom)

def create_dataset_income(transfrom):
    return SatDataset([income_task], image_root, transfrom)
