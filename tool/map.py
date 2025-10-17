import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define region
lon_min, lon_max = 73, 136
lat_min, lat_max = 3, 55

# Create map figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Set map extent
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Add base map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
ax.add_feature(cfeature.OCEAN, facecolor='#e0f0ff')

# Draw bounding box
from matplotlib.patches import Rectangle
rect = Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                 linewidth=2, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree())
ax.add_patch(rect)

# Add gridlines
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

plt.title('Region: 3°N–55°N, 73°E–136°E')
plt.show()