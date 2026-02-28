import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

class ShipDatabase:
    _instance = None
    _data = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ShipDatabase, cls).__new__(cls)
            cls._instance._load_data()
        return cls._instance

    def _load_data(self):
        project_root = Path(__file__).resolve().parent.parent.parent
        json_path = project_root / 'data' / 'processed_ships.json'
        
        if not json_path.exists():
             raise FileNotFoundError(f"Could not find processed_ships.json at {json_path}")
             
        with open(json_path, 'r') as f:
            self._data = json.load(f)
            
        # Index by ID for faster lookup
        self._data_by_id = {item['id']: item for item in self._data if 'id' in item}

    def get_boat_by_id(self, boat_id: str) -> Optional[Dict[str, Any]]:
        return self._data_by_id.get(str(boat_id))

    def get_all_ids(self):
        return list(self._data_by_id.keys())

def get_boat_radii(boat_id: str):
    """
    Returns (radii, length, width) for a given boat ID.
    Radii are equi-angular from -pi to pi.
    """
    db = ShipDatabase()
    boat = db.get_boat_by_id(boat_id)
    
    if not boat:
        raise ValueError(f"Boat ID {boat_id} not found in database.")
    
    radii = np.array(boat['radii'])
    
    return radii
