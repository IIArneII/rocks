import json, math
IN_JSON  = "sam2_masks/sam2_labels.json"
OUT_JSON = "sam2_masks/filtered_labels.json"

def keep(rec):
    A = rec["area"]
    if A < 400 or A > 0.2*rec["height"]*rec["width"]:
        return False
    w, h = rec["bbox"][2:]
    if max(w, h)/ (min(w, h)+1e-4) > 3.0:
        return False
    if rec["iou"] < 0.9:
        return False
    return True

recs = [r for r in json.load(open(IN_JSON)) if keep(r)]
json.dump(recs, open(OUT_JSON, "w"))
print("kept", len(recs), "masks (", len(recs)/len(json.load(open(IN_JSON)))*100, "% )")