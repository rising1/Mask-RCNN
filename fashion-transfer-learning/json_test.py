import json

f = open('E:\Fashion\\train\\via_data.json')

data = json.load(f)

#for i in data['_via_img_metadata']:
#    for j in data['_via_img_metadata'][i]['regions']:
#        print (j['shape_attributes']['all_points_x'],j['shape_attributes']['all_points_y'])

# data = list(data.values())  # don't need the dict keys

# The VIA tool saves images in the JSON even if they don't have any
# annotations. Skip unannotated images.
#for a in data['file_attributes']:
# annotations = [a for a in data if a['file_attributes']]
#for i in data:
#    print( i )
print(data['_via_settings'])
# objects = [s['caption'] for s in a['file_attributes']]  # altered


# data = data['_via_img_metadata']

f.close()

#with open('via_data_test_formatted.txt',"w") as write_file:
#    json.dump(annotations,write_file, indent=2)
#with open('E:\\Fashion\\spliced\\via_data_edited.json',"w") as write_file:
#    json.dump(data,write_file)