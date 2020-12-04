import sys
import folium
import numpy as np

m = folium.Map(location=[30.262561, -97.742319])

input_points, generated_points = np.load(sys.argv[1], allow_pickle=True)

for i, x in enumerate(input_points):
    folium.Marker(
        location=x,
        popup=str(i + 1),
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

for i, x in enumerate(generated_points):
    folium.Marker(
        location=x,
        popup=str(i + 1 + len(input_points)),
        icon=folium.Icon(color='green', icon='info-sign')
    ).add_to(m)

m.save(id_str[:len(id_str) - 4] + '.html')
