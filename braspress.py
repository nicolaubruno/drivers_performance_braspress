import numpy as np
import pandas as pd
from geopy import distance

class Braspress:
    @property
    def branches(self):
        return self._branches
    
    @property
    def settings(self):
        return self._settings
    
    @property
    def valid_lines(self):
        return self._valid_lines
    
    def __init__(self, settings: dict[str, str]):
        self._settings = settings
        self._branches = self._extract_braspress_branches()

    def _extract_braspress_branches(self) -> pd.DataFrame:
        raw_df: pd.DataFrame = pd.read_csv(self.settings["braspress_branches_path"], delimiter=",", dtype = str)
        clean_df: pd.DataFrame = (
            raw_df
            .rename(columns={"SIGLA": "id", "ENDEREÇO": "address", "LAT/LONG": "coordinates" })
            .set_index("id")
        )

        clean_df["coordinates"] = clean_df["coordinates"].str.split(", ", n=1)
        clean_df["coordinates"] = clean_df["coordinates"].transform(lambda coord: (float(coord[0].replace(',', '.')), float(coord[1].replace(',', '.'))))
        
        return clean_df

    def extract_valid_lines_with_stats(self, drivers_stats: pd.DataFrame) -> pd.Series:
        raw_valid_lines: pd.DataFrame = pd.read_csv(self.settings["braspress_valid_lines_path"], delimiter=",", dtype = str)
        valid_lines: pd.DataFrame = (
            raw_valid_lines
            .rename(columns={
                "Origem": "origin",
                "Destino": "destiny",
                "Org e Dst": "line_id",
                "Classificação de Linhas": "classification",
                "Linha": "line"
            })
        )

        valid_lines["line_id"] = valid_lines["line_id"].str.replace('-', '').astype(str)
        valid_lines["line_id"].drop_duplicates(inplace=True)

        valid_lines_with_stats = [] 

        for line_id, driver_stats in drivers_stats.groupby(by="line_id"):
            valid_lines_with_stats.append(pd.DataFrame({
                "line_id": [line_id],
                "avg_fuel_consumption_mean": [driver_stats["avg_fuel_consumption"].mean()],
                "avg_fuel_consumption_std": [driver_stats["avg_fuel_consumption"].std()],
            }))

        valid_lines_with_stats = pd.concat(valid_lines_with_stats).set_index("line_id")

        valid_lines_with_stats["avg_fuel_consumption_goal"] = valid_lines_with_stats.apply(
            lambda row: row["avg_fuel_consumption_mean"] if np.isnan(row["avg_fuel_consumption_std"]) else row["avg_fuel_consumption_mean"] + row["avg_fuel_consumption_std"]*self.settings["avg_fuel_consumption_goal_number_of_std"],
              axis = 1)
        
        valid_lines_with_stats.drop(columns=["avg_fuel_consumption_mean", "avg_fuel_consumption_std"], inplace=True)
        
        return (
            pd
            .merge(valid_lines["line_id"], valid_lines_with_stats, on="line_id", how="left")
            .set_index("line_id"))
            #.rename(columns={"avg_fuel_consumption": "avg_fuel_consumption_goal"}))
        
    def get_closest_branch(self, coord: tuple[float, float]) -> str:
        branches = self.branches
        branches["distance"] = branches["coordinates"].transform(lambda ref_coord: distance.geodesic(ref_coord, coord).km)

        closest_braspress = branches.loc[branches["distance"] < self.settings["braspress_proximity_threshold"]].index.values

        return closest_braspress[0] if len(closest_braspress) > 0 else None