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

        '''
        clean_df[["latitude", "longitude"]] = clean_df["coordinates"].str.split(", ", n=1, expand=True)
        clean_df["latitude"] = clean_df["latitude"].str.replace(",", ".").astype(float)
        clean_df["longitude"] = clean_df["longitude"].str.replace(",", ".").astype(float)
        '''

        clean_df["coordinates"] = clean_df["coordinates"].str.split(", ", n=1)
        clean_df["coordinates"] = clean_df["coordinates"].transform(lambda coord: (float(coord[0].replace(',', '.')), float(coord[1].replace(',', '.'))))

        #clean_df.drop(columns=["coordinates"], inplace=True)
        
        return clean_df

    def get_closest_branch(self, coord: tuple[float, float]) -> str:
        branches = self.branches
        branches["distance"] = branches["coordinates"].transform(lambda ref_coord: distance.geodesic(ref_coord, coord).km)

        closest_braspress = branches.loc[branches["distance"] < self.settings["braspress_proximity_threshold"]].index.values

        return closest_braspress[0] if len(closest_braspress) > 0 else None