// Global map variables
let map;
let marker;

// Initialize map
function initMap() {
    try {
        // Check if Leaflet is loaded
        if (typeof L === 'undefined') {
            console.error('Leaflet library not loaded!');
            document.getElementById('mapInfo').innerHTML = '⚠️ Map library failed to load. Please refresh the page.';
            return;
        }
        
        // Default center (Kampala, Uganda)
        const defaultLat = 0.3476;
        const defaultLng = 32.5825;
        
        // Create map
        map = L.map('map').setView([defaultLat, defaultLng], 13);
        
        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 19
        }).addTo(map);
        
        // Add click event to map
        map.on('click', function(e) {
            const lat = e.latlng.lat;
            const lng = e.latlng.lng;
            updateLocation(lat, lng);
        });
        
        // Set initial marker
        updateLocation(defaultLat, defaultLng);
    } catch (error) {
        console.error('Error initializing map:', error);
        document.getElementById('mapInfo').innerHTML = '⚠️ Error loading map. Please refresh the page.';
    }
}

// Update location with marker and coordinates
function updateLocation(lat, lng, locationName = null) {
    // Remove existing marker if any
    if (marker) {
        map.removeLayer(marker);
    }
    
    // Add new marker
    marker = L.marker([lat, lng], {
        draggable: true
    }).addTo(map);
    
    // Add popup to marker
    const popupContent = locationName 
        ? `<b>${locationName}</b><br>Lat: ${lat.toFixed(6)}<br>Lng: ${lng.toFixed(6)}`
        : `<b>Selected Location</b><br>Lat: ${lat.toFixed(6)}<br>Lng: ${lng.toFixed(6)}`;
    marker.bindPopup(popupContent).openPopup();
    
    // Update marker drag event
    marker.on('dragend', function(e) {
        const position = e.target.getLatLng();
        updateCoordinates(position.lat, position.lng);
        marker.bindPopup(`<b>Custom Location</b><br>Lat: ${position.lat.toFixed(6)}<br>Lng: ${position.lng.toFixed(6)}`).openPopup();
    });
    
    // Update coordinates
    updateCoordinates(lat, lng);
    
    // Center map on new location
    map.setView([lat, lng], map.getZoom());
}

// Update coordinate inputs
function updateCoordinates(lat, lng) {
    document.getElementById('latitude').value = lat.toFixed(6);
    document.getElementById('longitude').value = lng.toFixed(6);
    document.getElementById('mapInfo').innerHTML = `
        📍 <strong>Selected Location:</strong> 
        <span style="font-family: monospace;">Lat: ${lat.toFixed(6)}, Lng: ${lng.toFixed(6)}</span>
    `;
}

// Search for location
async function searchLocation(query) {
    if (!query.trim()) {
        showError('Please enter a location to search');
        return;
    }
    
    try {
        // Using Nominatim API for geocoding
        const response = await fetch(
            `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=1`
        );
        
        if (!response.ok) {
            throw new Error('Search failed');
        }
        
        const data = await response.json();
        
        if (data.length === 0) {
            showError('Location not found. Please try a different search term.');
            return;
        }
        
        const result = data[0];
        const lat = parseFloat(result.lat);
        const lng = parseFloat(result.lon);
        
        updateLocation(lat, lng, result.display_name);
        
        // Zoom to location
        map.setView([lat, lng], 15);
        
    } catch (error) {
        showError(`Search error: ${error.message}`);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Check if map element exists
    const mapElement = document.getElementById('map');
    if (!mapElement) {
        console.error('Map container element not found!');
        return;
    }
    
    // Initialize map with a small delay to ensure DOM is fully ready
    setTimeout(function() {
        initMap();
    }, 100);
    
    const submitBtn = document.getElementById('submit');
    const latitudeInput = document.getElementById('latitude');
    const longitudeInput = document.getElementById('longitude');
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    
    // Search button click event
    searchBtn.addEventListener('click', function() {
        const query = searchInput.value;
        searchLocation(query);
    });
    
    // Search on Enter key
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchBtn.click();
        }
    });

    submitBtn.addEventListener('click', async function() {
        const latitude = parseFloat(latitudeInput.value);
        const longitude = parseFloat(longitudeInput.value);

        // Validation
        if (isNaN(latitude) || isNaN(longitude)) {
            showError('Please select a location on the map or search for a place');
            return;
        }

        if (latitude < -90 || latitude > 90) {
            showError('Latitude must be between -90 and 90');
            return;
        }

        if (longitude < -180 || longitude > 180) {
            showError('Longitude must be between -180 and 180');
            return;
        }

        // Hide previous results and errors
        resultsDiv.classList.remove('show');
        errorDiv.classList.remove('show');
        loadingDiv.classList.add('show');
        submitBtn.disabled = true;

        try {
            const response = await fetch('/api/inference', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    latitude: latitude,
                    longitude: longitude
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Network response was not ok');
            }

            const result = await response.json();

            console.log('Inference result:', result);

            displayResults(result);
        } catch (error) {
            showError(`Error: ${error.message}`);
        } finally {
            loadingDiv.classList.remove('show');
            submitBtn.disabled = false;
        }
    });

    function displayResults(data) {
        document.getElementById('eatScoreNow').textContent = data.eat_score_now.toFixed(2);
        document.getElementById('eatScoreFuture').textContent = data.eat_score_future.toFixed(2);
        document.getElementById('external_pm25').textContent = data.external_pm25.toFixed(2);
        
        const currentAnomalyBadge = document.getElementById('currentAnomaly');
        currentAnomalyBadge.textContent = data.current_anomaly_detected ? 'Anomaly Detected' : 'Normal';
        currentAnomalyBadge.className = 'anomaly-badge ' + (data.current_anomaly_detected ? 'detected' : 'normal');
        
        const futureAnomalyBadge = document.getElementById('futureAnomaly');
        futureAnomalyBadge.textContent = data.future_anomaly_detected ? 'Anomaly Detected' : 'Normal';
        futureAnomalyBadge.className = 'anomaly-badge ' + (data.future_anomaly_detected ? 'detected' : 'normal');
        
        document.getElementById('coordinates').textContent = `${data.latitude}, ${data.longitude}`;
        
        resultsDiv.classList.add('show');
    }

    function showError(message) {
        errorDiv.textContent = message;
        errorDiv.classList.add('show');
        setTimeout(() => {
            errorDiv.classList.remove('show');
        }, 5000);
    }
});