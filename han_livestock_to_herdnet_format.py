import os
import xml.etree.ElementTree as ET
import numpy as np

sets = ['train', 'test']
annotation_files = []
possible_extensions = ['.xml']
output_files = ['{}_han_livestock_to_herdnet_format.csv'.format(s) for s in sets]

# Output file format: images,x,y,labels
for output_file in output_files:
    with open(output_file, 'w') as f:
        f.write('images,x,y,labels\n')

for root, dirs, files in os.walk('.'):
    # except for the data_preview folder
    if root == './data_preview':
        continue
    for file in files:
        if any(file.endswith(ext) for ext in possible_extensions):
            annotation_files.append(os.path.join(root, file))

image_sizes = []
for xml_file in annotation_files:
    xml_file = os.path.normpath(xml_file)
    parts = xml_file.split(os.sep)
    if 'train' in parts[0]:
        output_file = output_files[0]
        image_filename = parts[-1].replace('.xml', '.jpg')
    elif 'test' in parts[0]:
        output_file = output_files[1]
        image_filename = parts[-1].replace('.xml', '.JPG')
    else:
        raise ValueError(f'File {xml_file} is not in a train or test folder')
    
    # Parse the xml file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    image_sizes.append([width, height])
    
    # Write the image annotation to the output file
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name != 'unknown':
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            x_center = int((xmin + xmax) / 2)
            y_center = int((ymin + ymax) / 2)
            with open(output_file, 'a') as f:
                f.write(f'{image_filename},{x_center},{y_center},1\n') # 1 is the label for 'animal'

# Print some statistics
image_sizes = np.array(image_sizes)
print(f'Average image size: {np.mean(image_sizes, axis=0)}')
print(f'Minimum image size: {np.min(image_sizes, axis=0)}')
print(f'Maximum image size: {np.max(image_sizes, axis=0)}')
