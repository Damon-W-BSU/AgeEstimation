import os
import pandas as pd
from PIL import Image

images = os.listdir('facial_data/UTKFace/')
paths = []
ages = []
np = 0
for i in images:
    image = Image.open('facial_data/UTKFace/' + i)
    age = int(i[0:i.index('_')])
    if age >= 5 and age <= 25:
        new_image = image.resize((1,1))
        new_image = new_image.resize((224,224))
        new_image.save('facial_data/processed_images/' + i)
        paths.append(i)
        ages.append(int(i[0:i.index('_')]))
        np += 1
    if (np % 1000 == 0) and (np > 0):
        print('images processed: ', end="")
        print(np)

print('Images in new data:', np)

d ={'path': paths, 'age': ages}
df = pd.DataFrame(data=d)
df["path"] = ('facial_data/processed_images/' + df['path'])
print(df)
