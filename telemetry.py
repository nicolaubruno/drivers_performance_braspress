import numpy as np
import pandas as pd
from geopy import distance

import importlib
import braspress

importlib.reload(braspress)
import braspress


class Telemetry:
    @property
    def raw_telemetry_data(self):
        return self._raw_telemetry_data
    
    @property
    def telemetry_data(self):
        return self._telemetry_data
    
    @property
    def settings(self):
        return self._settings
    
    @property
    def braspress(self):
        return self._braspress
    
    @property
    def max_lines_to_read(self):
        return self._max_lines_to_read
    
    def __init__(self, settings: dict[str, str], max_lines_to_read: int = None):
        self._settings = settings
        self._max_lines_to_read = max_lines_to_read
        self._braspress = braspress.Braspress(self.settings)

        self._raw_telemetry_data = self._extract_raw_telemetry_data()
        self._telemetry_data = self._clean_raw_telemetry_data()

    def _clean_raw_telemetry_data(self) -> pd.DataFrame:
        # Rename columns
        columns_rename = {        
            "Frota": "fleet",
            "CPF": "driver_id",
            "Início": "startDate",
            "Fim": "endDate",
            "Lat/Long inicial": "initial_coord",
            "Lat/Long final": "final_coord",
            "Distância (Km)": "distance",
            "Total Litros Consumido": "fuel"
        }

        if self.max_lines_to_read is not None:
            clean_data = self.raw_telemetry_data.iloc[:self.max_lines_to_read][columns_rename.keys()].astype(str).rename(columns=columns_rename)
        
        else:
            clean_data = self.raw_telemetry_data[columns_rename.keys()].astype(str).rename(columns=columns_rename)

        # Convert brazilian decimal to Python float
        clean_data["distance"] = clean_data["distance"].apply(lambda d: d.replace(",", ".")).astype(float)
        clean_data["fuel"] = clean_data["fuel"].apply(lambda d: d.replace(",", ".")).astype(float)
        
        # Split INITIAL coordinates into two columns: latitude and longitude
        clean_data["initial_coord"] = clean_data["initial_coord"].str.split(", ", n=1)
        clean_data["initial_coord"] = clean_data["initial_coord"].transform(lambda coord: (float(coord[0].replace(',', '.')), float(coord[1].replace(',', '.'))))
        
        # Split FINAL coordinates into two columns: latitude and longitude
        clean_data["final_coord"] = clean_data["final_coord"].str.split(", ", n=1)
        clean_data["final_coord"] = clean_data["final_coord"].transform(lambda coord: (float(coord[0].replace(',', '.')), float(coord[1].replace(',', '.'))))

        # Remove non-defined drivers
        clean_data = clean_data[clean_data["driver_id"] != '-']

        # Remove small distances
        clean_data = clean_data[clean_data["distance"] >= self.settings["min_displacement"]]
        
        # Check proximity with a Braspress branch
        clean_data["initial_coord_close_to_brasprass"] = clean_data["initial_coord"].transform(lambda coord: self.braspress.get_closest_branch(coord))
        clean_data["final_coord_close_to_brasprass"] = clean_data["final_coord"].transform(lambda coord: self.braspress.get_closest_branch(coord))

        return clean_data.reset_index(drop=True)

    def _extract_raw_telemetry_data(self) -> pd.DataFrame:
        raw_telemetry_data = []
        for date in self.settings["dates"]:
            path: str = f"{self.settings["telemetry_dir"]}/{date[0]:04}-{date[1]:02}.csv"
            raw_telemetry_data.append(pd.read_csv(path, delimiter=";", dtype = str))
        
        return pd.concat(raw_telemetry_data)

    def get_fleet_driver_groups(self):
        fleet_driver_groups = [group for _, group in self.telemetry_data.groupby(by=["fleet", "driver_id"], sort=False)]
        valid_trajectories = pd.DataFrame({
            "fleet": pd.Series(dtype=str),
            "driver_id": pd.Series(dtype=str),
            "line_id": pd.Series(dtype=str),
            "avg_fuel_consumption": pd.Series(dtype=float)
        })

        for fleet_driver_group in fleet_driver_groups:
            start_traj_idx: int = -1
            start_traj_branch: str = None
            end_traj_idx: int = -1
            end_traj_branch: str = None

            start_traj: pd.DataFrame = fleet_driver_group.loc[~fleet_driver_group["initial_coord_close_to_brasprass"].isnull()]
            if start_traj.shape[0] > 0:
                start_traj_idx = start_traj.index.values[0]
                start_traj_branch = start_traj["initial_coord_close_to_brasprass"].loc[start_traj_idx]

            end_traj: pd.DataFrame = fleet_driver_group.loc[~fleet_driver_group["final_coord_close_to_brasprass"].isnull()]
            if end_traj.shape[0] > 0:
                end_traj_idx = end_traj.index.values[0]
                end_traj_branch = end_traj["final_coord_close_to_brasprass"].loc[end_traj_idx]
                if(start_traj_branch == end_traj_branch):
                    end_traj_idx = -1

            if start_traj_idx > -1 and end_traj_idx > -1:
                distance: float = fleet_driver_group["distance"].loc[start_traj_idx:end_traj_idx].sum()
                fuel: float = fleet_driver_group["fuel"].loc[start_traj_idx:end_traj_idx].sum()
                avg_fuel_consumption = (distance / fuel) if fuel > 0 else np.nan

                new_valid_traj = pd.DataFrame({
                    "fleet": [start_traj["fleet"].iloc[0]],
                    "driver_id": [start_traj["driver_id"].iloc[0]],
                    "avg_fuel_consumption": [avg_fuel_consumption],
                    "line_id": [start_traj["initial_coord_close_to_brasprass"].iloc[0] + end_traj["final_coord_close_to_brasprass"].iloc[0]]
                })

                valid_trajectories = pd.concat([valid_trajectories, new_valid_traj], ignore_index=True)

            if end_traj_idx >= 0 and end_traj_idx < fleet_driver_group.index.values[-1]:
                fleet_driver_group_remaining = fleet_driver_group.loc[(end_traj_idx+1):]
                start_traj_idx: int = -1
                start_traj_branch: str = None
                end_traj_idx: int = -1
                end_traj_branch: str = None

                start_traj: pd.DataFrame = fleet_driver_group_remaining.loc[~fleet_driver_group_remaining["initial_coord_close_to_brasprass"].isnull()]
                if start_traj.shape[0] > 0:
                    start_traj_idx = start_traj.index.values[0]
                    start_traj_branch = start_traj["initial_coord_close_to_brasprass"].loc[start_traj_idx]

                end_traj: pd.DataFrame = fleet_driver_group_remaining.loc[~fleet_driver_group_remaining["final_coord_close_to_brasprass"].isnull()]
                if end_traj.shape[0] > 0:
                    end_traj_idx = end_traj.index.values[0]
                    end_traj_branch = end_traj["final_coord_close_to_brasprass"].loc[end_traj_idx]
                    if(start_traj_branch == end_traj_branch):
                        end_traj_idx = -1

                if start_traj_idx > -1 and end_traj_idx > -1:
                    distance: float = fleet_driver_group["distance"].loc[start_traj_idx:end_traj_idx].sum()
                    fuel: float = fleet_driver_group["fuel"].loc[start_traj_idx:end_traj_idx].sum()
                    avg_fuel_consumption = (distance / fuel) if fuel > 0 else np.nan

                    new_valid_traj = pd.DataFrame({
                        "fleet": [start_traj["fleet"].iloc[0]],
                        "driver_id": [start_traj["driver_id"].iloc[0]],
                        "avg_fuel_consumption": [avg_fuel_consumption],
                        "line_id": [start_traj["initial_coord_close_to_brasprass"].iloc[0] + end_traj["final_coord_close_to_brasprass"].iloc[0]]
                    })

                    valid_trajectories = pd.concat([valid_trajectories, new_valid_traj], ignore_index=True)
        
        return valid_trajectories

    def get_driver_stats(self):
        drivers_stats = pd.concat([group.drop(columns=["fleet"]) for _, group in self.get_fleet_driver_groups().groupby(by=["driver_id"], sort=False)])
        valid_lines_with_stats = braspress.Braspress(self.settings).extract_valid_lines_with_stats(drivers_stats)
        drivers_stats = pd.merge(drivers_stats, valid_lines_with_stats, how="left", on="line_id")

        drivers_stats_avg = []
        for driver_id, driver_stats in drivers_stats.groupby(by="driver_id"):
            stats = pd.DataFrame({
                "driver_id": [driver_id],
                "avg_fuel_consumption": [driver_stats["avg_fuel_consumption"].mean()],
                "avg_fuel_consumption_goal": [driver_stats["avg_fuel_consumption_goal"].mean()],
            })
            stats["reach_fuel_goal"] = stats.apply(lambda row: np.nan if np.isnan(row["avg_fuel_consumption_goal"]) else row["avg_fuel_consumption"] > row["avg_fuel_consumption_goal"], axis = 1)

            drivers_stats_avg.append(stats)
        
        return pd.concat(drivers_stats_avg).set_index("driver_id").sort_index(ascending=False)