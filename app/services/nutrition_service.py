NUTRITION_DB = {
    'apple': {'calories': 95, 'protein': 0.5, 'carbs': 25, 'fats': 0.3, 'sugar': 19, 'fiber': 4.4, 'portion': 180},
    'banana': {'calories': 105, 'protein': 1.3, 'carbs': 27, 'fats': 0.3, 'sugar': 14, 'fiber': 3.1, 'portion': 118},
    'sandwich': {'calories': 250, 'protein': 12, 'carbs': 30, 'fats': 10, 'sugar': 4, 'fiber': 2, 'portion': 150},
    'orange': {'calories': 62, 'protein': 1.2, 'carbs': 15, 'fats': 0.2, 'sugar': 12, 'fiber': 3.1, 'portion': 130},
    'broccoli': {'calories': 55, 'protein': 3.7, 'carbs': 11, 'fats': 0.6, 'sugar': 2.5, 'fiber': 5.2, 'portion': 148},
    'carrot': {'calories': 41, 'protein': 0.9, 'carbs': 10, 'fats': 0.2, 'sugar': 4.7, 'fiber': 2.8, 'portion': 100},
    'hot dog': {'calories': 290, 'protein': 10, 'carbs': 24, 'fats': 17, 'sugar': 4, 'fiber': 1, 'portion': 100},
    'pizza': {'calories': 285, 'protein': 12, 'carbs': 36, 'fats': 10, 'sugar': 3, 'fiber': 2.5, 'portion': 107},
    'donut': {'calories': 195, 'protein': 2, 'carbs': 22, 'fats': 11, 'sugar': 11, 'fiber': 0.8, 'portion': 50},
    'cake': {'calories': 235, 'protein': 3, 'carbs': 35, 'fats': 10, 'sugar': 20, 'fiber': 0.5, 'portion': 64},
    'indian_thali': {'calories': 850, 'protein': 22, 'carbs': 130, 'fats': 25, 'sugar': 8, 'fiber': 15, 'portion': 500},
    'unknown_food_item': {'calories': 150, 'protein': 5, 'carbs': 15, 'fats': 5, 'sugar': 5, 'fiber': 2, 'portion': 100}
}

def estimate_nutrition(detections):
    results = {}
    for d in detections:
        name = d.get('name')
        conf = d.get('confidence')
        nutri_info = NUTRITION_DB.get(name, NUTRITION_DB['unknown_food_item'])
        
        results[name] = {
            'calories': nutri_info['calories'],
            'protein': nutri_info['protein'],
            'carbs': nutri_info['carbs'],
            'fats': nutri_info['fats'],
            'sugar': nutri_info['sugar'],
            'fiber': nutri_info['fiber'],
            'portion': nutri_info['portion'],
            'confidence': conf
        }
        
    return results
