import json

def process_coco_annotations(input_path, output_path, categories_to_remove, new_order):
    # Load original annotations
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Create quick lookup for categories
    category_map = {cat['id']: cat for cat in data['categories']}
    
    # Remove specified categories and validate new order
    remaining_cats = [cat for cat in data['categories'] if cat['name'] not in categories_to_remove]
    remaining_names = {cat['name'] for cat in remaining_cats}
    
    # Verify new order contains exactly the remaining categories
    if set(new_order) != remaining_names:
        raise ValueError("New order must contain exactly the remaining categories after removal")
    
    # Reindex categories according to new order
    new_categories = []
    category_id_mapping = {}
    for new_id, cat_name in enumerate(new_order, start=1):
        old_cat = next(c for c in remaining_cats if c['name'] == cat_name)
        category_id_mapping[old_cat['id']] = new_id
        new_categories.append({
            'id': new_id,
            'name': cat_name,
            'supercategory': old_cat['supercategory']
        })
    
    # Update annotations
    new_annotations = []
    for ann in data['annotations']:
        if ann['category_id'] in category_id_mapping:
            ann['category_id'] = category_id_mapping[ann['category_id']]
            new_annotations.append(ann)
    
    # Create new data structure
    new_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'categories': new_categories,
        'images': data['images'],
        'annotations': new_annotations
    }
    
    # Save modified annotations
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)

process_coco_annotations(
    input_path='data/coco/auto.json',
    output_path='data/coco/auto_modified.json',
    categories_to_remove=['region', 'ladle_points', 'ladle_handle_points', 'ladle', 'ladle_handle'],
    new_order=['rock']
)