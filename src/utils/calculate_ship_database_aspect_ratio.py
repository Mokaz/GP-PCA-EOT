import json
import os
import numpy as np

def main():
    json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed_ships.json')
    
    with open(json_path, 'r') as f:
        ships = json.load(f)
        
    aspect_ratios = []
    
    for ship in ships:
        # Check criteria: is_boat == 1 and is_kayak == 0
        if ship.get('is_boat') == 1 and ship.get('is_kayak') == 0:
            length = ship.get('original_length_m')
            width = ship.get('original_width_m')
            
            if length is not None and width is not None and width > 0:
                aspect_ratios.append(length / width)
                
    if aspect_ratios:
        avg_ratio = np.mean(aspect_ratios)
        std_ratio = np.std(aspect_ratios)
        median_ratio = np.median(aspect_ratios)
        min_ratio = np.min(aspect_ratios)
        max_ratio = np.max(aspect_ratios)
        
        print(f"Processed {len(aspect_ratios)} ships (is_boat=1, is_kayak=0).")
        print(f"--------------------------------------------------")
        print(f"Average Aspect Ratio (L/W) : {avg_ratio:.4f}")
        print(f"Median Aspect Ratio        : {median_ratio:.4f}")
        print(f"Standard Deviation         : {std_ratio:.4f}")
        print(f"Min Aspect Ratio           : {min_ratio:.4f}")
        print(f"Max Aspect Ratio           : {max_ratio:.4f}")
    else:
        print("No valid ships found matching the criteria.")

if __name__ == '__main__':
    main()
