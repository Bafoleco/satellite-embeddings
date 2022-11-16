import pandas as pd
import tasks 

def collect_temps(id_source):
    id_source_df = pd.read_csv(id_source)


    temps = id_source_df["ID"].copy()

    print(temps)

    for i in range(len(id_source_df)):
        id = id_source_df.iloc[i, id_source_df.columns.get_loc("ID")]
        print(id)

        # 

    



collect_temps(tasks.treecover_task.csv_file)