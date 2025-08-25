import pandas as pd
import gpxpy
from fastkml import kml
from fitparse import FitFile
import tcxparser
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


class GPSConverter:
    def __init__(self, file):
        self.convert_to_parquet(file, Path(file).stem + ".parquet")

    def gpx_to_df(self, file_path):
        with open(file_path, 'r') as f:
            gpx = gpxpy.parse(f)
        data = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    data.append({
                        'latitude': point.latitude,
                        'longitude': point.longitude
                    })
        return pd.DataFrame(data)

    def kml_to_df(self, file_path):
        with open(file_path, 'rb') as f:
            doc = f.read()
        k_obj = kml.KML()
        k_obj.from_string(doc)
        features = list(k_obj.features())
        data = []
        for feature in features:
            for placemark in feature.features():
                if placemark.geometry:
                    coords = placemark.geometry.coords[0]
                    data.append({
                        'latitude': coords[1],
                        'longitude': coords[0]
                    })
        return pd.DataFrame(data)

    def fit_to_df(self, file_path):
        fitfile = FitFile(file_path)
        data = []
        for record in fitfile.get_messages('record'):
            row = {}
            for field in record:
                if field.name in ['position_lat', 'position_long']:
                    # Convertir si nécessaire en degrés
                    row['latitude' if field.name=='position_lat' else 'longitude'] = field.value
            if row:
                data.append(row)
        return pd.DataFrame(data)

    def tcx_to_df(self, file_path):
        tcx = tcxparser.TCXParser(file_path)
        data = []
        if tcx.latitude and tcx.longitude:
            for lat, lon in zip(tcx.latitude, tcx.longitude):
                data.append({'latitude': lat, 'longitude': lon})
        return pd.DataFrame(data)

    def convert_to_parquet(self, input_file, output_file):
        ext = Path(input_file).suffix.lower()
        if ext == '.gpx':
            df = self.gpx_to_df(input_file)
        elif ext == '.kml':
            df = self.kml_to_df(input_file)
        elif ext == '.fit':
            df = self.fit_to_df(input_file)
        elif ext == '.tcx':
            df = self.tcx_to_df(input_file)
        else:
            raise ValueError(f"Format non supporté: {ext}")

        # Conserver uniquement les colonnes demandées
        df = df[['latitude', 'longitude']]
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file)
        print(f"Converti {input_file} -> {output_file}")


class ParquetLoader:
    def __init__(self):
        self.data = None

    def load(self, file_path):
        """
        Charge un fichier .parquet et le stocke dans self.data (DataFrame pandas)
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
        self.data = pd.read_parquet(file_path)
        print(f"{file_path} chargé avec {len(self.data)} lignes.")
        return self.data

    def save(self, data, file_path):
        """
        Sauvegarde un DataFrame pandas dans un fichier .parquet
        """
        table = pa.Table.from_pandas(data)
        pq.write_table(table, file_path)
        print(f"{file_path} sauvegardé avec {len(data)} lignes.")