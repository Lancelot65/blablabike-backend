import pandas as pd
import folium

class TraceMap:
    def __init__(self, zoom_start=13):
        self.traces = []  # liste des DataFrames de traces
        self.zoom_start = zoom_start
        self.map = None

    def add_trace(self, df, name=None, color="blue"):
        """
        Ajouter une trace à la carte.
        df : pandas DataFrame avec colonnes 'latitude' et 'longitude'
        name : nom du tracé (optionnel)
        color : couleur du tracé
        """
        if not {'latitude', 'longitude'}.issubset(df.columns):
            raise ValueError("Le DataFrame doit contenir les colonnes 'latitude' et 'longitude'.")
        self.traces.append({'df': df, 'name': name, 'color': color})

    def add_marker(self, latitude, longitude, popup_text=None, tooltip_text=None, color="red", icon="info-sign"):
        """
        Ajouter un marqueur individuel à la carte.
        
        Parameters:
        - latitude : float, latitude du point
        - longitude : float, longitude du point
        - popup_text : str, texte affiché lors du clic (optionnel)
        - tooltip_text : str, texte affiché au survol (optionnel)
        - color : str, couleur du marqueur
        - icon : str, icône du marqueur
        """
        marker_info = {
            'lat': latitude,
            'lon': longitude,
            'popup': popup_text,
            'tooltip': tooltip_text,
            'color': color,
            'icon': icon
        }
        
        # Créer une liste de marqueurs si elle n'existe pas
        if not hasattr(self, 'markers'):
            self.markers = []
        
        self.markers.append(marker_info)

    def create_map(self):
        """
        Créer la carte et ajouter toutes les traces et marqueurs.
        """
        if not self.traces:
            raise ValueError("Aucune trace à afficher.")
        
        # Centrer la carte sur la moyenne de tous les points
        all_lats = pd.concat([t['df']['latitude'] for t in self.traces])
        all_lons = pd.concat([t['df']['longitude'] for t in self.traces])
        center_lat = all_lats.mean()
        center_lon = all_lons.mean()
        
        self.map = folium.Map(location=[center_lat, center_lon], zoom_start=self.zoom_start)
        
        # Ajouter les traces
        for t in self.traces:
            points = list(zip(t['df']['latitude'], t['df']['longitude']))
            folium.PolyLine(points, color=t['color'], weight=5, opacity=0.7, tooltip=t['name']).add_to(self.map)
            
            # Ajouter début et fin
            folium.Marker(points[0], tooltip=f"{t['name']} Début" if t['name'] else "Début").add_to(self.map)
            folium.Marker(points[-1], tooltip=f"{t['name']} Fin" if t['name'] else "Fin").add_to(self.map)
        
        # Ajouter les marqueurs individuels
        if hasattr(self, 'markers'):
            for marker in self.markers:
                folium.Marker(
                    location=[marker['lat'], marker['lon']],
                    popup=marker['popup'],
                    tooltip=marker['tooltip'],
                    icon=folium.Icon(color=marker['color'], icon=marker['icon'])
                ).add_to(self.map)
        
        return self.map

    def save(self, filename="map.html"):
        """
        Sauvegarder la carte en HTML.
        """
        if self.map is None:
            raise ValueError("La carte n'a pas encore été créée. Utilisez create_map() d'abord.")
        self.map.save(filename)

    def add_curve(self, points, color="green", weight=3, opacity=0.8, tooltip=None, is_point=False, marker_kwargs=None):
        """
        Ajouter une courbe (ligne) ou des points à la carte.
        points : liste de tuples (lat, lon) ou DataFrame avec colonnes 'latitude'/'longitude'.
        color : couleur de la courbe ou des points.
        weight : épaisseur de la ligne.
        opacity : opacité de la ligne.
        tooltip : texte affiché au survol.
        is_point : si True, ajoute des points (Marker), sinon une PolyLine.
        marker_kwargs : dict d'options supplémentaires pour les Marker.
        """
        if self.map is None:
            raise ValueError("La carte n'a pas encore été créée. Utilisez create_map() d'abord.")

        import folium
        import pandas as pd

        # Si DataFrame, extraire les points
        if isinstance(points, pd.DataFrame):
            points = list(zip(points['latitude'], points['longitude']))

        if is_point:
            marker_kwargs = marker_kwargs or {}
            for lat, lon in points:
                folium.Marker(
                    location=[lat, lon],
                    icon=folium.Icon(color=color, **marker_kwargs),
                    tooltip=tooltip
                ).add_to(self.map)
        else:
            folium.PolyLine(points, color=color, weight=weight, opacity=opacity, tooltip=tooltip).add_to(self.map)