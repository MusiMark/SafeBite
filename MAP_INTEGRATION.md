# Map Integration - Complete Guide

## Overview
The API frontend now includes an interactive map that allows users to search for locations and automatically send coordinates (latitude/longitude) to the API for air quality predictions.

## Features

### 1. **Interactive Map**
- **Library**: Leaflet.js (lightweight, open-source)
- **Map Provider**: OpenStreetMap tiles
- **Default Location**: Kampala, Uganda (0.3476, 32.5825)
- **Height**: 400px, rounded corners, responsive

### 2. **Location Search**
- Search bar for finding places by name
- Powered by OpenStreetMap Nominatim geocoding API (free, no API key required)
- Examples: "Kampala, Uganda", "Nairobi, Kenya", "Times Square, New York"

### 3. **User Interaction Methods**

#### Method 1: Search by Place Name
1. Type a location in the search box
2. Click "Search" or press Enter
3. Map zooms to location and places marker
4. Coordinates automatically filled

#### Method 2: Click on Map
1. Click anywhere on the map
2. Marker placed at that location
3. Coordinates automatically updated

#### Method 3: Drag the Marker
1. Click and hold the marker
2. Drag to new location
3. Coordinates update in real-time

## User Interface

### Clean, Search-Focused Design
- **No manual coordinate entry** - hidden input fields store values
- **Search bar** - Primary method for location selection
- **Map display** - Visual selection and confirmation
- **Info box** - Shows selected coordinates in readable format
- **Large "Analyze Air Quality" button** - Clear call-to-action

### Layout Flow
```
┌─────────────────────────────────────┐
│  🔍 Search Location                 │
│  [Search box] [Search button]       │
│  ┌───────────────────────────────┐  │
│  │       Interactive Map         │  │
│  │     (Click or drag marker)    │  │
│  └───────────────────────────────┘  │
│  📍 Selected: Lat: X, Lng: Y        │
│      [Analyze Air Quality]          │
│  ───────────────────────────────    │
│  📊 Analysis Results (after query)  │
└─────────────────────────────────────┘
```

## How It Works

### Workflow
1. **User Action**: Search/Click/Drag on map
2. **Coordinate Extraction**: Latitude and longitude captured
3. **Auto-Population**: Hidden input fields updated
4. **Display Update**: Info box shows coordinates
5. **User Clicks**: "Analyze Air Quality" button
6. **API Request**: Coordinates sent to `/api/inference` endpoint
7. **Results Display**: Air quality predictions shown

### API Integration
```javascript
POST /api/inference
Content-Type: application/json

{
    "latitude": 0.347600,
    "longitude": 32.582500
}
```

## Files Modified

### 1. `src/static/index.html`
**Added:**
- Leaflet CSS library link (CDN)
- Leaflet JavaScript library link (CDN)
- Search input and button elements
- Map container div (`<div id="map">`)
- Hidden coordinate inputs
- Enhanced CSS for map styling

**Removed:**
- Visible latitude/longitude input boxes
- Manual coordinate entry fields

### 2. `src/static/app.js`
**Added Functions:**
- `initMap()` - Initializes map with default location, adds tiles, sets up events
- `updateLocation(lat, lng, locationName)` - Updates marker position and coordinates
- `updateCoordinates(lat, lng)` - Updates hidden input fields and info display
- `searchLocation(query)` - Handles geocoding search via Nominatim API

**Event Listeners:**
- Search button click
- Search input Enter key
- Map click event
- Marker drag event

## Technical Details

### Dependencies
- **Leaflet.js**: v1.9.4 (CDN)
- **OpenStreetMap Tiles**: Free, no API key
- **Nominatim Geocoding**: Free (1 request/second limit)

### Browser Compatibility
- Modern browsers (Chrome, Firefox, Edge, Safari)
- Requires JavaScript enabled
- Requires internet connection for CDN and tiles

### CSS Styling
```css
#map {
    height: 400px;
    width: 100%;
    border-radius: 10px;
    border: 2px solid #e0e0e0;
}

.map-info {
    background-color: #f0f7ff;
    padding: 15px;
    text-align: center;
    color: #1976d2;
}
```

## Usage Guide

### For End Users

1. **Search for a Location**
   - Type city name, address, or landmark
   - Press Enter or click Search button
   - Examples: "Kampala", "Makerere University", "Central Park"

2. **Verify Location**
   - Check the marker position on map
   - Read coordinates in info box
   - Drag marker to fine-tune if needed

3. **Analyze Air Quality**
   - Click "Analyze Air Quality" button
   - Wait for results (loading spinner appears)
   - View predictions and air quality scores

### For Developers

#### Running the Server
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Start server
python run.py

# Access at
http://localhost:8000
```

#### Testing the Map
1. Open browser to `http://localhost:8000`
2. Open DevTools (F12)
3. Check Console for errors
4. Check Network tab for resource loading

#### API Endpoint
The map sends coordinates to your existing inference endpoint:
```python
@router.post("/inference")
async def get_inference(request: LocationRequest):
    # request.latitude and request.longitude
    # are automatically populated from map
    ...
```

## Customization

### Change Default Location
```javascript
// In app.js, modify initMap()
const defaultLat = YOUR_LATITUDE;
const defaultLng = YOUR_LONGITUDE;
map = L.map('map').setView([defaultLat, defaultLng], ZOOM_LEVEL);
```

### Change Map Height
```css
/* In index.html styles */
#map {
    height: 500px; /* Change from 400px */
}
```

### Custom Marker Icon
```javascript
const customIcon = L.icon({
    iconUrl: '/static/custom-marker.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});
marker = L.marker([lat, lng], { icon: customIcon });
```

### Different Map Tiles
```javascript
// Satellite view example
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri'
}).addTo(map);
```

## Troubleshooting

### Map Not Loading - Grey Box
**Problem**: Map container appears but tiles don't load
**Solution**: 
- Check internet connection
- Verify Leaflet CSS is loading (F12 → Network tab)
- Check browser console for errors

### "L is not defined" Error
**Problem**: JavaScript error in console
**Solution**:
- Ensure Leaflet.js loads before app.js
- Check script order in index.html:
  ```html
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="/static/app.js"></script>
  ```

### Search Not Working
**Problem**: Location search returns no results
**Solution**:
- Be more specific (include country name)
- Check Nominatim API rate limit (1 req/sec)
- Verify internet connection
- Try different search terms

### Map Container Not Found
**Problem**: Console shows "map container not found"
**Solution**:
- Verify `<div id="map">` exists in HTML
- Check JavaScript loads after DOM ready
- Already fixed with setTimeout in code

### Coordinates Not Updating
**Problem**: Clicking map doesn't update coordinates
**Solution**:
- Check browser console for JavaScript errors
- Verify hidden input fields exist (`id="latitude"`, `id="longitude"`)
- Check `updateCoordinates()` function is defined

## Best Practices

### For Production

1. **Add Loading States**
   - Show spinner while map initializes
   - Disable buttons until map loads

2. **Error Handling**
   - Catch and display geocoding errors
   - Handle network failures gracefully

3. **Rate Limiting**
   - Implement debouncing for search
   - Cache common searches
   - Consider paid geocoding service for high traffic

4. **Performance**
   - Lazy load Leaflet if not immediately visible
   - Optimize marker updates
   - Minimize map redraws

5. **Accessibility**
   - Add ARIA labels to map controls
   - Provide keyboard navigation
   - Include alt text for icons

## Future Enhancements

### Planned Features
- [ ] Geolocation - Detect user's current location
- [ ] Air quality heatmap overlay on map
- [ ] Multiple location comparison
- [ ] Save favorite locations
- [ ] Location history/recent searches
- [ ] Autocomplete for search
- [ ] Custom marker icons for air quality levels
- [ ] Street view integration
- [ ] Mobile app with native maps

### Advanced Features
- [ ] Real-time air quality data visualization
- [ ] Historical data timeline slider
- [ ] Weather overlay
- [ ] Traffic layer
- [ ] Nearby sensors/monitoring stations
- [ ] Export map as image
- [ ] Share location via URL

## Security Considerations

1. **Input Validation**
   - Validate coordinates before sending to API
   - Sanitize search queries
   - Rate limit API requests

2. **API Security**
   - Use HTTPS in production
   - Implement CORS properly
   - Add authentication if needed

3. **Privacy**
   - Don't store location without consent
   - Comply with GDPR/privacy laws
   - Allow users to clear location data

## Performance Metrics

### Target Metrics
- Map load time: < 2 seconds
- Search response: < 1 second
- Marker update: Instant (< 100ms)
- API request: < 3 seconds

### Optimization Tips
- Use CDN for static assets
- Enable browser caching
- Compress images and icons
- Minimize JavaScript bundle
- Use async/defer for scripts

## Support & Resources

### Documentation
- [Leaflet Documentation](https://leafletjs.com/reference.html)
- [OpenStreetMap](https://www.openstreetmap.org/)
- [Nominatim API Docs](https://nominatim.org/release-docs/latest/api/Search/)

### Common Issues
If you encounter issues:
1. Check browser console (F12)
2. Verify network requests in DevTools
3. Test with different browsers
4. Clear browser cache
5. Check Nominatim service status

---

## Quick Reference

### Start Server
```bash
.\venv\Scripts\activate
python run.py
```

### Access Application
```
http://localhost:8000
```

### Test Location Search
Try searching for:
- "Kampala, Uganda"
- "Nairobi, Kenya"  
- "Lagos, Nigeria"
- "New York, USA"

### Verify Map Works
✅ Map tiles load
✅ Default marker appears
✅ Search finds locations
✅ Click places marker
✅ Drag moves marker
✅ Coordinates update
✅ "Analyze" sends to API

---

**Status**: ✅ Complete and Operational
**Last Updated**: October 30, 2025
**Version**: 1.0
