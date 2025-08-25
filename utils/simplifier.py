import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
import warnings

class GPSSimplifier:
    """
    Simplificateur GPS optimisé utilisant l'algorithme Ramer-Douglas-Peucker
    avec projection métrique et optimisations NumPy/Numba.
    """
    
    EARTH_RADIUS_M = 6371000.0
    
    def __init__(self, epsilon_meters: float = 10.0):
        """
        Initialise le simplificateur GPS optimisé.
        
        Args:
            epsilon_meters: Tolérance de simplification en mètres
            use_numba: Utiliser la compilation JIT Numba pour les performances
        """
        self.epsilon_meters = epsilon_meters
    
    def latlon_to_xy(self, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
        """
        Convertit lat/lon (degrés) vers coordonnées planaires x,y (mètres).
        
        Args:
            latitudes: Array des latitudes en degrés
            longitudes: Array des longitudes en degrés
            
        Returns:
            Array Nx2 des coordonnées x,y en mètres
        """
        latitudes = np.asarray(latitudes, dtype=np.float64)
        longitudes = np.asarray(longitudes, dtype=np.float64)
        
        if len(latitudes) == 0:
            return np.empty((0, 2), dtype=np.float64)
            
        if len(latitudes) == 0:
            return np.empty((0, 2), dtype=np.float64)
            
        # Latitude de référence pour la projection équirectangulaire
        lat0_rad = np.deg2rad(np.mean(latitudes))
        cos_lat0 = np.cos(lat0_rad)
        
        # Conversion vectorisée
        lat_rad = np.deg2rad(latitudes)
        lon_rad = np.deg2rad(longitudes)
        
        x = self.EARTH_RADIUS_M * lon_rad * cos_lat0
        y = self.EARTH_RADIUS_M * lat_rad
        
        return np.column_stack((x, y))
    

    def _point_segment_distance_numpy(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """Version NumPy du calcul de distance point-segment."""
        ab = b - a
        ap = p - a
        ab_len2 = np.dot(ab, ab)
        
        if ab_len2 == 0:
            return np.linalg.norm(ap)
        
        t = np.dot(ap, ab) / ab_len2
        t = np.clip(t, 0.0, 1.0)
        nearest = a + t * ab
        
        return np.linalg.norm(p - nearest)
    

    def _rdp_indices_numpy(self, points_xy: np.ndarray, epsilon: float) -> np.ndarray:
        """Version NumPy de l'algorithme RDP."""
        n = len(points_xy)
        if n <= 2:
            return np.arange(n)
        
        stack = [(0, n - 1)]
        keep = np.zeros(n, dtype=bool)
        keep[0] = True
        keep[-1] = True
        
        while stack:
            i, j = stack.pop()
            if j - i <= 1:
                continue
                
            a = points_xy[i]
            b = points_xy[j]
            max_dist = -1.0
            max_idx = -1
            
            # Vectorisation partielle pour trouver le point le plus éloigné
            segment_points = points_xy[i+1:j]
            distances = np.array([
                self._point_segment_distance_numpy(point, a, b) 
                for point in segment_points
            ])
            
            if len(distances) > 0:
                max_idx_rel = np.argmax(distances)
                max_dist = distances[max_idx_rel]
                max_idx = i + 1 + max_idx_rel
            
            if max_dist > epsilon:
                keep[max_idx] = True
                stack.append((i, max_idx))
                stack.append((max_idx, j))
        
        return np.nonzero(keep)[0]
    
    def rdp_indices(self, points_xy: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Applique l'algorithme Ramer-Douglas-Peucker et retourne les indices à conserver.
        
        Args:
            points_xy: Array Nx2 des coordonnées planaires
            epsilon: Tolérance en mètres
            
        Returns:
            Array des indices des points à conserver
        """
        if len(points_xy) == 0:
            return np.array([], dtype=int)
            
        points_xy = np.asarray(points_xy, dtype=np.float64)
        
        return self._rdp_indices_numpy(points_xy, epsilon)
    
    def simplify(self, 
                 data: Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]], 
                 epsilon_meters: Optional[float] = None) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """
        Simplifie une trace GPS.
        
        Args:
            data: DataFrame avec colonnes 'latitude'/'longitude' ou tuple (latitudes, longitudes)
            epsilon_meters: Tolérance en mètres (utilise self.epsilon_meters si None)
            
        Returns:
            Données simplifiées dans le même format que l'entrée
        """
        epsilon = epsilon_meters if epsilon_meters is not None else self.epsilon_meters
        
        if isinstance(data, pd.DataFrame):
            return self._simplify_dataframe(data, epsilon)
        else:
            return self._simplify_arrays(data, epsilon)
    
    def _simplify_dataframe(self, df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
        """Simplifie un DataFrame."""
        if not {'latitude', 'longitude'}.issubset(df.columns):
            raise ValueError("DataFrame must contain 'latitude' and 'longitude' columns")
        
        if len(df) <= 2:
            return df.copy()
        
        latitudes = df['latitude'].to_numpy()
        longitudes = df['longitude'].to_numpy()
        
        # Conversion vers coordonnées planaires
        points_xy = self.latlon_to_xy(latitudes, longitudes)
        
        # Application RDP
        indices = self.rdp_indices(points_xy, epsilon)
        
        return df.iloc[indices].reset_index(drop=True)
    
    def _simplify_arrays(self, arrays: Tuple[np.ndarray, np.ndarray], epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
        """Simplifie des arrays NumPy."""
        latitudes, longitudes = arrays
        latitudes = np.asarray(latitudes)
        longitudes = np.asarray(longitudes)
        
        if len(latitudes) <= 2:
            return latitudes.copy(), longitudes.copy()
        
        # Conversion vers coordonnées planaires
        points_xy = self.latlon_to_xy(latitudes, longitudes)
        
        # Application RDP
        indices = self.rdp_indices(points_xy, epsilon)
        
        return latitudes[indices], longitudes[indices]