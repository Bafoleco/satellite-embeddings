import pandas as pd
import tasks 
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Normals, Monthly, Daily, Stations, Hourly

def collect_temps(id_source):
    id_source_df = pd.read_csv(id_source)

    temps = id_source_df[["ID", "lat", "lon"]].copy()

    avg_temps = []
    lat_loc = temps.columns.get_loc("lat")
    lon_loc = temps.columns.get_loc("lon")
    for i in range(len(id_source_df)):
        id = temps.iloc[i, temps.columns.get_loc("ID")]
        lat = temps.iloc[i, lat_loc]
        lon = temps.iloc[i, lon_loc]


        start = datetime(2018, 1, 1)
        end = datetime(2018, 12, 31)

        point = Point(lat, lon)
        data = Hourly(point, start, end)
        data = data.fetch()

        # print(data)
        
        # stations = Stations()
        # stations = stations.nearby(lat, lon)
        
        # station = stations.fetch(1)
        # print(station)

    

        print(data)

        if i > 20:
            break

        avg_temp = data["temp"].mean()

        print("Average temp for ", str(lat) + ", " + str(lon), " is ", avg_temp)

        avg_temps.append(avg_temp)




    # # add temps col
    # temps.insert(len(temps.columns), "temps", avg_temps)

    # # save to csv
    # temps.to_csv("temps.csv", index=False)
    



collect_temps(tasks.elevation_task.csv_file)