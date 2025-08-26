import time
from typing import List, Tuple, Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy import distance as geopy_distance
from geopy.point import Point


class Find_with_zone:
    """
    Trouve les fichiers d'itinéraires (parquet contenant 'latitude'/'longitude')
    qui passent près de deux villes données, plus la distances du parcours est grande plus ça recherche loin.

    Principales améliorations :
    - géocodage robuste avec cache,
    - recherche du point le plus proche sur la trace pour chaque ville,
    - vérification de l'ordre (ville1 avant ville2),
    - plot amélioré (itinéraire, villes, cercles géodésiques).

    TODO
    - si la distance entre les deux point sur la carte est trop faible ducoup il parcours moins de 30km ensemble alors  c ciao
    """

    def __init__(
        self,
        files: List[str],
        cercle_ratio: float = 0.1,
        num_points_circle: int = 120,
        geolocator: Optional[Nominatim] = None,
        geocode_delay: float = 1.0,
    ):
        self.files = files
        self.cercle_ratio = cercle_ratio
        self.num_points_circle = num_points_circle
        self.geocode_delay = geocode_delay
        self.geolocator = geolocator or Nominatim(user_agent="route_finder")
        self._geocode_cache = {}  # cache pour éviter de re-géocoder
        self.available_files = []

    # --- Géocodage / utilitaires ---
    def _geocode(self, city: Union[str, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """
        city peut être :
          - un nom de ville (str) -> on utilise Nominatim
          - un tuple (lat, lon) -> renvoyé tel quel
        Retourne (lat, lon) ou None si introuvable.
        """
        if isinstance(city, tuple) and len(city) == 2:
            return city

        if city in self._geocode_cache:
            return self._geocode_cache[city]

        try:
            loc = self.geolocator.geocode(city, exactly_one=True, timeout=10)
            time.sleep(self.geocode_delay)  # respecter rate-limit
            if loc:
                coords = (loc.latitude, loc.longitude)
                self._geocode_cache[city] = coords
                return coords
            else:
                print(f"[Géocodage] Ville '{city}' non trouvée.")
                self._geocode_cache[city] = None
                return None
        except Exception as e:
            print(f"[Géocodage] Erreur pour '{city}': {e}")
            return None

    @staticmethod
    def _km_between(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return geopy_distance.distance(a, b).km

    @staticmethod
    def route_length_km(df: pd.DataFrame) -> float:
        """Distance totale approximative de l'itinéraire (somme des segments)."""
        coords = list(zip(df['latitude'], df['longitude']))
        if len(coords) < 2:
            return 0.0
        total = 0.0
        for i in range(len(coords) - 1):
            total += geopy_distance.distance(coords[i], coords[i + 1]).km
        return total

    @staticmethod
    def find_nearest_point_index(df: pd.DataFrame, point: Tuple[float, float]) -> Tuple[int, float]:
        """
        Trouve l'indice du point dans df le plus proche de 'point'.
        Retourne (indice, distance_km).
        """
        best_idx = -1
        best_dist = float('inf')
        for i, row in df[['latitude', 'longitude']].iterrows():
            d = geopy_distance.distance((row['latitude'], row['longitude']), point).km
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx, best_dist

    def generate_circle(self, center: Tuple[float, float], radius_km: float) -> pd.DataFrame:
        """
        Retourne un DataFrame de coordonnées formant un cercle géodésique autour de center.
        """
        lat0, lon0 = center
        points = []
        for bearing in np.linspace(0, 360, self.num_points_circle, endpoint=False):
            dest = geopy_distance.distance(kilometers=radius_km).destination(Point(lat0, lon0), bearing)
            points.append((dest.latitude, dest.longitude))
        return pd.DataFrame(points, columns=['latitude', 'longitude'])

    # --- Process / filtrage ---
    def process(self, city1: Union[str, Tuple[float, float]], city2: Union[str, Tuple[float, float]],
                require_order: bool = True, min_radius_km: float = 1.0):
        """
        Cherche dans self.files les fichiers qui passent près de city1 et city2.
        - city1 / city2 peuvent être noms ou tuples (lat, lon).
        - require_order: si True, exige que l'index du point proche de city1 soit < index du point proche de city2.
        - min_radius_km: rayon minimal (en km) si cercle_ratio*distance trop petit.
        Résultat dans self.available_files comme une liste de dicts :
          {'file': path, 'idx_city1': int, 'dist_city1_km': float, 'idx_city2': int, 'dist_city2_km': float, 'route_km': float}
        """
        self.available_files = []
        city1_coords = self._geocode(city1)
        city2_coords = self._geocode(city2)

        if city1_coords is None or city2_coords is None:
            print("[Process] Géocodage échoué pour l'une des villes — arrêt.")
            return

        for file in self.files:
            try:
                df = pd.read_parquet(file)
            except Exception as e:
                print(f"[Process] Impossible de lire '{file}': {e}")
                continue

            if 'latitude' not in df.columns or 'longitude' not in df.columns:
                print(f"[Process] '{file}' ne contient pas 'latitude'/'longitude'.")
                continue

            route_km = self.route_length_km(df)
            radius_km = max(self.cercle_ratio * route_km, min_radius_km)

            idx1, d1 = self.find_nearest_point_index(df, city1_coords)
            idx2, d2 = self.find_nearest_point_index(df, city2_coords)

            # Condition : points proches des villes
            within1 = d1 <= radius_km
            within2 = d2 <= radius_km

            # Condition d'ordre
            order_ok = (idx1 < idx2) if require_order else True

            if within1 and within2 and order_ok:
                self.available_files.append({
                    'file': file,
                    'route_km': route_km,
                    'radius_km': radius_km,
                    'city1': {'coords': city1_coords, 'idx': idx1, 'dist_km': d1},
                    'city2': {'coords': city2_coords, 'idx': idx2, 'dist_km': d2},
                })
                print(f"[Match] '{file}' ✓  (route {route_km:.1f} km, rayon {radius_km:.2f} km)")
            else:
                print(
                    f"[NoMatch] '{file}' — city1 dist {d1:.2f} km (within={within1}), "
                    f"city2 dist {d2:.2f} km (within={within2}), order_ok={order_ok}"
                )

        if not self.available_files:
            print("[Process] Aucun fichier compatible trouvé.")

    # --- Plot ---
    def plot(self, show_all_matches: bool = True, figsize: Tuple[int, int] = (8, 8), savepath: Optional[str] = None):
        """
        Affiche les routes trouvées (self.available_files). Si show_all_matches=True, trace chaque fichier trouvé.
        Ajoute les villes, les cercles et les points les plus proches.
        """
        if not self.available_files:
            print("[Plot] Aucun résultat à afficher. Lancez d'abord process(...).")
            return

        for item in self.available_files:
            file = item['file']
            df = pd.read_parquet(file)

            plt.figure(figsize=figsize)
            plt.plot(df['longitude'], df['latitude'], '-o', markersize=3, linewidth=1, label='Itinéraire')

            # villes et cercles
            c1 = item['city1']['coords']
            c2 = item['city2']['coords']
            r = item['radius_km']

            # cercles (DataFrame)
            circ1 = self.generate_circle(c1, r)
            circ2 = self.generate_circle(c2, r)
            plt.plot(circ1['longitude'], circ1['latitude'], '--', linewidth=1, label=f'cercle {r:.1f}km (ville 1)')
            plt.plot(circ2['longitude'], circ2['latitude'], '--', linewidth=1, label=f'cercle {r:.1f}km (ville 2)')

            # marquer les villes
            plt.scatter([c1[1]], [c1[0]], marker='*', s=120, label='Ville 1', zorder=5)
            plt.scatter([c2[1]], [c2[0]], marker='*', s=120, label='Ville 2', zorder=5)
            plt.annotate("Ville 1", xy=(c1[1], c1[0]), xytext=(5, 5), textcoords='offset points')
            plt.annotate("Ville 2", xy=(c2[1], c2[0]), xytext=(5, 5), textcoords='offset points')

            # marquer les points les plus proches sur la route
            idx1 = item['city1']['idx']
            idx2 = item['city2']['idx']
            p1 = (df.loc[idx1, 'latitude'], df.loc[idx1, 'longitude'])
            p2 = (df.loc[idx2, 'latitude'], df.loc[idx2, 'longitude'])
            plt.plot([p1[1]], [p1[0]], 'o', markersize=8, label='Point proche ville1')
            plt.plot([p2[1]], [p2[0]], 'o', markersize=8, label='Point proche ville2')
            plt.title(f"Itinéraire: {file}\ndistance totale ≈ {item['route_km']:.1f} km — rayon {r:.2f} km")
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.axis('equal')
            plt.legend()
            plt.grid(alpha=0.3)

            if savepath:
                plt.savefig(savepath)
                print(f"[Plot] Sauvegardé: {savepath}")

            plt.show()


rf = RouteFinder(["a_simplified.parquet", "b_simplified.parquet"])
rf.process("Coubert", "Paris", require_order=True)
# Voir résultats :
print(rf.available_files)  # détails utiles
rf.plot()  # affiche les routes trouvées avec villes + cercles
