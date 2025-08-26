import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcule la distance entre deux points GPS en utilisant la formule de Haversine.
    Retourne la distance en kilomètres.
    """
    # Convertir les degrés en radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Différences
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Formule de Haversine
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # Rayon de la Terre en km
    r = 6371
    
    return r * c

def extraire_depart_arrivee(df):
    """
    Extrait les coordonnées de départ et d'arrivée d'un DataFrame GPX.
    Suppose que les colonnes sont 'latitude' et 'longitude'.
    """
    if df.empty:
        return None, None, None, None
    
    # Premier point (départ)
    depart_lat = df.iloc[0]['latitude']
    depart_lon = df.iloc[0]['longitude']
    
    # Dernier point (arrivée)
    arrivee_lat = df.iloc[-1]['latitude']
    arrivee_lon = df.iloc[-1]['longitude']
    
    return depart_lat, depart_lon, arrivee_lat, arrivee_lon

def verifier_proximite(fichier1_parquet, fichier2_parquet, rayon_km=15):
    """
    Vérifie si les points de départ et d'arrivée de deux traces GPX sont proches.
    
    Args:
        fichier1_parquet (str): Chemin vers le premier fichier Parquet
        fichier2_parquet (str): Chemin vers le deuxième fichier Parquet
        rayon_km (float): Rayon de proximité en kilomètres (défaut: 15km)
    
    Returns:
        dict: Résultats de la comparaison
    """
    try:
        # Charger les deux fichiers Parquet
        df1 = pd.read_parquet(fichier1_parquet)
        df2 = pd.read_parquet(fichier2_parquet)
        
        print(f"Trace 1: {len(df1)} points chargés")
        print(f"Trace 2: {len(df2)} points chargés")
        
        # Extraire les points de départ et d'arrivée
        depart1_lat, depart1_lon, arrivee1_lat, arrivee1_lon = extraire_depart_arrivee(df1)
        depart2_lat, depart2_lon, arrivee2_lat, arrivee2_lon = extraire_depart_arrivee(df2)
        
        if None in [depart1_lat, depart1_lon, arrivee1_lat, arrivee1_lon]:
            print("Erreur: Impossible d'extraire les coordonnées de la trace 1")
            return None
            
        if None in [depart2_lat, depart2_lon, arrivee2_lat, arrivee2_lon]:
            print("Erreur: Impossible d'extraire les coordonnées de la trace 2")
            return None
        
        # Calculer les distances
        distance_departs = haversine_distance(depart1_lat, depart1_lon, depart2_lat, depart2_lon)
        distance_arrivees = haversine_distance(arrivee1_lat, arrivee1_lon, arrivee2_lat, arrivee2_lon)
        
        # Vérifier la proximité
        departs_proches = distance_departs <= rayon_km
        arrivees_proches = distance_arrivees <= rayon_km
        
        # Préparer les résultats
        resultats = {
            'trace1': {
                'depart': (depart1_lat, depart1_lon),
                'arrivee': (arrivee1_lat, arrivee1_lon)
            },
            'trace2': {
                'depart': (depart2_lat, depart2_lon),
                'arrivee': (arrivee2_lat, arrivee2_lon)
            },
            'distances': {
                'departs_km': round(distance_departs, 2),
                'arrivees_km': round(distance_arrivees, 2)
            },
            'proximite': {
                'departs_proches': departs_proches,
                'arrivees_proches': arrivees_proches,
                'rayon_km': rayon_km
            }
        }
        
        # Afficher les résultats
        print("\n" + "="*50)
        print("RÉSULTATS DE LA COMPARAISON")
        print("="*50)
        
        print(f"\nTrace 1:")
        print(f"  Départ: {depart1_lat:.6f}, {depart1_lon:.6f}")
        print(f"  Arrivée: {arrivee1_lat:.6f}, {arrivee1_lon:.6f}")
        
        print(f"\nTrace 2:")
        print(f"  Départ: {depart2_lat:.6f}, {depart2_lon:.6f}")
        print(f"  Arrivée: {arrivee2_lat:.6f}, {arrivee2_lon:.6f}")
        
        print(f"\nDistances calculées:")
        print(f"  Entre les départs: {distance_departs:.2f} km")
        print(f"  Entre les arrivées: {distance_arrivees:.2f} km")
        
        print(f"\nProximité (rayon de {rayon_km} km):")
        status_depart = "✓ OUI" if departs_proches else "✗ NON"
        status_arrivee = "✓ OUI" if arrivees_proches else "✗ NON"
        
        print(f"  Départs proches: {status_depart}")
        print(f"  Arrivées proches: {status_arrivee}")
        
        if departs_proches and arrivees_proches:
            print(f"\n🎯 Les deux traces ont des départs ET des arrivées proches!")
        elif departs_proches or arrivees_proches:
            print(f"\n⚠️  Les traces ont partiellement des points proches.")
        else:
            print(f"\n❌ Les traces n'ont ni départs ni arrivées proches.")
            
        return resultats
        
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")
        return None

# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacez ces chemins par vos vrais fichiers
    fichier1 = "a.parquet"
    fichier2 = "b.parquet"
    
    # Vérifier la proximité avec un rayon de 15km
    resultats = verifier_proximite(fichier1, fichier2, rayon_km=15)
    
    if resultats:
        # Vous pouvez aussi accéder aux données programmatiquement
        print(f"\nAccès programmatique aux résultats:")
        print(f"Distance entre départs: {resultats['distances']['departs_km']} km")
        print(f"Distance entre arrivées: {resultats['distances']['arrivees_km']} km")
        print(f"Départs proches: {resultats['proximite']['departs_proches']}")
        print(f"Arrivées proches: {resultats['proximite']['arrivees_proches']}")