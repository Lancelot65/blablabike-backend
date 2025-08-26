from utils import TraceMap
import numpy as np
from geopy.distance import geodesic
from utils import ParquetLoader, GPSSimplifier, GPSConverter


# ----------------- paramètres (modifie ici) -----------------
PARQUET_PATH = "b_simplified.parquet"
TRACEMAP_ZOOM = 10
LINE_POINTS = 200            # points pour tracer la droite
CONE_ANGLE_DEG = 100          # ouverture des cônes
CONE_LENGTH_FACTOR = 7.0     # longueur des cônes relative à la ligne centrale
NUM_CONE_POINTS = 100         # résolution des cônes
# Ces valeurs seront recalculées automatiquement selon la distance souhaitée
FIRST_CONE_START_FRAC = 0.5  # sera remplacé par la position à 40km avant l'arrivée
SECOND_CONE_START_FRAC = 1.0 # fraction pour le 2e cône (1.0 = fin), peut être <1
RECT_START_FRAC = 0.5        # sera remplacé par la position à 40km avant l'arrivée  
RECT_END_FRAC = -.8        # fin du rectangle (NÉGATIF = extension au-delà du DÉBUT de la ligne)
RECT_WIDTH_FACTOR = 0.2      # largeur du rectangle relative à la longueur de la ligne
COLORS = {"axis":"red","cone1":"black","cone2":"blue","rect":"green"}
# ------------------------------------------------------------

def calculate_gpx_distance(lats, lons):
    """Calcule la distance totale du tracé GPX en kilomètres."""
    total_distance = 0.0
    for i in range(1, len(lats)):
        point1 = (lats[i-1], lons[i-1])
        point2 = (lats[i], lons[i])
        total_distance += geodesic(point1, point2).kilometers
    return total_distance

def calculate_line_distance(x0, y0, x1, y1):
    """Calcule la distance de la droite rouge en kilomètres."""
    point1 = (y0, x0)  # (lat, lon)
    point2 = (y1, x1)  # (lat, lon)
    return geodesic(point1, point2).kilometers

def get_position_km_before_end(x_start, y_start, x_end, y_end, distance_km_before):
    """
    Calcule la position sur la droite rouge à distance_km_before de la fin.
    
    Args:
        x_start, y_start: coordonnées du début de la droite (lon, lat)
        x_end, y_end: coordonnées de la fin de la droite (lon, lat)
        distance_km_before: distance en km avant l'arrivée
    
    Returns:
        tuple: (lon, lat, fraction) - position et fraction le long de la ligne
        None si distance_km_before > longueur totale
    """
    total_distance = calculate_line_distance(x_start, y_start, x_end, y_end)
    
    if distance_km_before >= total_distance:
        print(f"Attention: distance demandée ({distance_km_before:.3f}km) >= distance totale ({total_distance:.3f}km)")
        return None
    
    # Fraction depuis le début (1.0 = fin, 0.0 = début)
    fraction_from_start = 1.0 - (distance_km_before / total_distance)
    
    # Interpolation linéaire
    x_pos = x_start + fraction_from_start * (x_end - x_start)
    y_pos = y_start + fraction_from_start * (y_end - y_start)
    
    return x_pos, y_pos, fraction_from_start

# Chargement
df = ParquetLoader().load(PARQUET_PATH)
mape = TraceMap(TRACEMAP_ZOOM)
mape.add_trace(df)

# Coordonnées (lon, lat)
xs = df["longitude"].values
ys = df["latitude"].values

# Calcul de la distance du tracé GPX
gpx_distance_km = calculate_gpx_distance(ys, xs)

# Régression linéaire y = a*x + b
a, b = np.polyfit(xs, ys, 1)

def project_point(xp, yp, a, b):
    """Projette (xp,yp) sur la droite y = a*x + b -> renvoie (x_proj, y_proj)."""
    if a == 0:
        return xp, b
    x_proj = (xp + a * yp - a * b) / (a**2 + 1)
    y_proj = a * x_proj + b
    return x_proj, y_proj

# Projection du début et de la fin pour définir la ligne centrale
x0_proj, y0_proj = project_point(xs[0], ys[0], a, b)
x1_proj, y1_proj = project_point(xs[-1], ys[-1], a, b)

# Calcul de la distance de la droite rouge
red_line_distance_km = calculate_line_distance(x0_proj, y0_proj, x1_proj, y1_proj)

# Affichage des distances calculées
print(f"=== DISTANCES CALCULÉES ===")
print(f"Distance du tracé GPX : {gpx_distance_km:.3f} km")
print(f"Distance de la droite rouge : {red_line_distance_km:.3f} km")
print(f"Ratio (droite/GPX) : {red_line_distance_km/gpx_distance_km:.3f}")
print(f"==============================")

# Calcul de la position à 40km avant l'arrivée pour positionner les formes
distance_avant_arrivee = 40.0  # km
position_40km = get_position_km_before_end(x0_proj, y0_proj, x1_proj, y1_proj, distance_avant_arrivee)

if position_40km:
    x_40km, y_40km, fraction_40km = position_40km
    print(f"\n=== POSITION À {distance_avant_arrivee}KM AVANT L'ARRIVÉE ===")
    print(f"Longitude : {x_40km:.6f}")
    print(f"Latitude : {y_40km:.6f}")
    print(f"Fraction le long de la ligne : {fraction_40km:.3f}")
    print(f"Distance depuis le début : {fraction_40km * red_line_distance_km:.3f} km")
    print(f"Distance jusqu'à l'arrivée : {(1-fraction_40km) * red_line_distance_km:.3f} km")
    print("=" * 45)
    
    # Mise à jour des paramètres pour utiliser la position calculée
    FIRST_CONE_START_FRAC = fraction_40km  # Premier cône commence à 40km avant l'arrivée
    RECT_START_FRAC = fraction_40km        # Rectangle commence aussi à 40km avant l'arrivée
    
    print(f"\n>>> Premier cône et rectangle positionnés à la fraction {fraction_40km:.3f} de la ligne")
    print(f">>> (soit à {distance_avant_arrivee}km avant l'arrivée)")
else:
    print(f"\nErreur : impossible de placer les formes à {distance_avant_arrivee}km avant l'arrivée")
    print(">>> Utilisation des paramètres par défaut")
    FIRST_CONE_START_FRAC = 0.5
    RECT_START_FRAC = 0.5

# Ligne centrale (liste de (lat, lon) pour TraceMap)
x_line = np.linspace(x0_proj, x1_proj, LINE_POINTS)
y_line = a * x_line + b
axis_points = list(zip(y_line, x_line))  # (lat, lon)

# utilitaires géométriques
def unit_vector_and_perp(x_coords, y_coords):
    dx = x_coords[-1] - x_coords[0]
    dy = y_coords[-1] - y_coords[0]
    length = np.hypot(dx, dy)
    if length == 0:
        return (1, 0), (0, 1), 1.0
    ux, uy = dx/length, dy/length
    px, py = -uy, ux  # perpendiculaire unitaire
    return (ux, uy), (px, py), length

(dir_vec, perp_vec, line_len) = unit_vector_and_perp(x_line, y_line)
dir_x, dir_y = dir_vec
perp_x, perp_y = perp_vec
half_angle = np.radians(CONE_ANGLE_DEG/2)

def make_cone(x_line, y_line, start_frac=1.0, direction_sign=-1, length_factor=1.0, num_points=NUM_CONE_POINTS):
    """
    Crée un cône polygonal centré sur la ligne :
      start_frac : fraction le long de la ligne où commence la base du cône (0..1)
      direction_sign: -1 pour aller vers l'arrière, +1 pour vers l'avant
      length_factor: multiplicateur de la longueur du cône par rapport à la ligne
    Renvoie une liste de (lat, lon) pts formant un polygone.
    """
    i0 = int(np.clip(start_frac * (len(x_line)-1), 0, len(x_line)-1))
    sx, sy = x_line[i0], y_line[i0]
    cone_len = line_len * length_factor
    pts = []
    # côté gauche (aller)
    for i in range(num_points):
        t = i/(num_points-1)
        cx = sx + direction_sign * t * cone_len * dir_x
        cy = sy + direction_sign * t * cone_len * dir_y
        width = t * np.tan(half_angle) * cone_len
        pts.append((cy + width*perp_y, cx + width*perp_x))
    # côté droit (retour)
    for i in reversed(range(num_points)):
        t = i/(num_points-1)
        cx = sx + direction_sign * t * cone_len * dir_x
        cy = sy + direction_sign * t * cone_len * dir_y
        width = t * np.tan(half_angle) * cone_len
        pts.append((cy - width*perp_y, cx - width*perp_x))
    return pts

def make_rectangle(x_line, y_line, start_frac=0.5, end_frac=0.02, width_factor=0.2, extend_beyond_line=True):
    """
    Rectangle le long de la ligne entre deux fractions (renvoie liste de (lat,lon)).
    Si extend_beyond_line=True et end_frac<0, le rectangle s'étend au-delà du DÉBUT de la ligne.
    """
    i0 = int(np.clip(start_frac * (len(x_line)-1), 0, len(x_line)-1))
    sx, sy = x_line[i0], y_line[i0]
    
    if extend_beyond_line and end_frac < 0:
        # Calculer un point au-delà du DÉBUT de la ligne (direction inverse)
        # Direction de la ligne (du début vers la fin)
        dx = x_line[-1] - x_line[0]
        dy = y_line[-1] - y_line[0]
        line_length = np.hypot(dx, dy)
        
        if line_length > 0:
            # Normaliser la direction (vers la fin)
            dir_x = dx / line_length
            dir_y = dy / line_length
            
            # Distance d'extension (proportionnelle à la longueur de la ligne)
            extension_distance = abs(end_frac) * line_length
            
            # Point étendu au-delà du DÉBUT (direction inverse)
            ex = x_line[0] - extension_distance * dir_x
            ey = y_line[0] - extension_distance * dir_y
        else:
            ex, ey = x_line[0], y_line[0]
    else:
        # Mode normal : end_frac dans les limites de la ligne
        i1 = int(np.clip(end_frac * (len(x_line)-1), 0, len(x_line)-1))
        ex, ey = x_line[i1], y_line[i1]
    
    width = line_len * width_factor
    rect = [
        (sy - width*perp_y, sx - width*perp_x),
        (sy + width*perp_y, sx + width*perp_x),
        (ey + width*perp_y, ex + width*perp_x),
        (ey - width*perp_y, ex - width*perp_x),
        (sy - width*perp_y, sx - width*perp_x),
    ]
    return rect

# Créer formes
cone1 = make_cone(x_line, y_line, start_frac=FIRST_CONE_START_FRAC, direction_sign=-1, length_factor=CONE_LENGTH_FACTOR)
cone2 = make_cone(x_line, y_line, start_frac=SECOND_CONE_START_FRAC, direction_sign=+1, length_factor=CONE_LENGTH_FACTOR)
rectangle = make_rectangle(x_line, y_line, start_frac=RECT_START_FRAC, end_frac=RECT_END_FRAC, width_factor=RECT_WIDTH_FACTOR, extend_beyond_line=True)

# Construire la carte
mape.create_map()
mape.add_curve(axis_points, color=COLORS["axis"], weight=3, opacity=0.8, tooltip="Axe central")
mape.add_curve(cone1, color=COLORS["cone1"], weight=2, opacity=0.7, tooltip="Cône 1")
mape.add_curve(cone2, color=COLORS["cone2"], weight=2, opacity=0.7, tooltip="Cône 2")
mape.add_curve(rectangle, color=COLORS["rect"], weight=2, opacity=0.7, tooltip="Rectangle")
mape.save()

print("Carte générée — modifie les paramètres en tête de fichier pour ajuster placement/taille.")