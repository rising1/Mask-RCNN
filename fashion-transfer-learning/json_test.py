import json

f = open('fashion/train/via_data.json')

data = json.load(f)

for i in data['_via_img_metadata']:
    for j in data['_via_img_metadata'][i]['regions']:
        print (j['shape_attributes']['all_points_x'],j['shape_attributes']['all_points_y'])

f.close()

#with open('via_data_formatted.txt',"w") as write_file:
#    json.dump(data,write_file, indent=2)
