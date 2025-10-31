// Global map variables
let map;
let marker;

// Uganda-specific places for autocomplete
const ugandaPlaces = [
    { name: "Kampala", region: "Central", lat: 0.3476, lon: 32.5825 },
    { name: "Entebbe", region: "Central", lat: 0.0560, lon: 32.4634 },
    { name: "Jinja", region: "Eastern", lat: 0.4244, lon: 33.2041 },
    { name: "Mbarara", region: "Western", lat: -0.6107, lon: 30.6575 },
    { name: "Gulu", region: "Northern", lat: 2.7746, lon: 32.2995 },
    { name: "Lira", region: "Northern", lat: 2.2399, lon: 32.8994 },
    { name: "Mbale", region: "Eastern", lat: 1.0827, lon: 34.1754 },
    { name: "Kasese", region: "Western", lat: 0.1838, lon: 30.0832 },
    { name: "Masaka", region: "Central", lat: -0.3388, lon: 31.7349 },
    { name: "Fort Portal", region: "Western", lat: 0.6621, lon: 30.2758 },
    { name: "Hoima", region: "Western", lat: 1.4332, lon: 31.3521 },
    { name: "Soroti", region: "Eastern", lat: 1.7149, lon: 33.6111 },
    { name: "Arua", region: "Northern", lat: 3.0193, lon: 30.9108 },
    { name: "Kabale", region: "Western", lat: -1.2488, lon: 29.9895 },
    { name: "Tororo", region: "Eastern", lat: 0.6931, lon: 34.1807 },
    { name: "Mukono", region: "Central", lat: 0.3531, lon: 32.7553 },
    { name: "Mityana", region: "Central", lat: 0.4175, lon: 32.0228 },
    { name: "Wakiso", region: "Central", lat: 0.4044, lon: 32.4594 },
    { name: "Kawempe", region: "Kampala", lat: 0.3905, lon: 32.5644 },
    { name: "Makindye", region: "Kampala", lat: 0.2825, lon: 32.5998 },
    { name: "Nakawa", region: "Kampala", lat: 0.3327, lon: 32.6182 },
    { name: "Rubaga", region: "Kampala", lat: 0.2996, lon: 32.5536 },
    { name: "Ntinda", region: "Kampala", lat: 0.3533, lon: 32.6189 },
    { name: "Kololo", region: "Kampala", lat: 0.3262, lon: 32.5988 },
    { name: "Nakasero", region: "Kampala", lat: 0.3219, lon: 32.5789 },
    { name: "Bugolobi", region: "Kampala", lat: 0.3114, lon: 32.6131 },
    { name: "Naguru", region: "Kampala", lat: 0.3352, lon: 32.6062 },
    { name: "Wandegeya", region: "Kampala", lat: 0.3336, lon: 32.5691 }
];

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
    const suggestionsDiv = document.getElementById('autocompleteSuggestions');
    
    let currentFocus = -1;
    
    // Autocomplete functionality for Uganda locations
    if (searchInput && suggestionsDiv) {
        searchInput.addEventListener('input', function() {
            const value = this.value.toLowerCase();
            suggestionsDiv.innerHTML = '';
            currentFocus = -1;
            
            if (!value || value.length < 2) {
                suggestionsDiv.style.display = 'none';
                return;
            }
            
            // Filter Uganda places
            const matches = ugandaPlaces.filter(place => 
                place.name.toLowerCase().includes(value) ||
                place.region.toLowerCase().includes(value)
            ).slice(0, 8); // Limit to 8 suggestions
            
            if (matches.length === 0) {
                suggestionsDiv.style.display = 'none';
                return;
            }
            
            matches.forEach(place => {
                const div = document.createElement('div');
                div.className = 'autocomplete-suggestion';
                
                const matchIndex = place.name.toLowerCase().indexOf(value);
                let displayName;
                
                if (matchIndex !== -1) {
                    const beforeMatch = place.name.substring(0, matchIndex);
                    const matchText = place.name.substring(matchIndex, matchIndex + value.length);
                    const afterMatch = place.name.substring(matchIndex + value.length);
                    displayName = `${beforeMatch}<strong>${matchText}</strong>${afterMatch}`;
                } else {
                    displayName = place.name;
                }
                
                div.innerHTML = `
                    <i class="fas fa-map-marker-alt"></i>
                    <span class="suggestion-text">${displayName}</span>
                    <span class="suggestion-type">${place.region}</span>
                `;
                
                div.addEventListener('click', function() {
                    selectPlace(place);
                });
                
                suggestionsDiv.appendChild(div);
            });
            
            suggestionsDiv.style.display = 'block';
        });
        
        // Keyboard navigation for autocomplete
        searchInput.addEventListener('keydown', function(e) {
            const suggestions = suggestionsDiv.getElementsByClassName('autocomplete-suggestion');
            
            if (e.keyCode === 40) { // Down arrow
                currentFocus++;
                addActive(suggestions);
                e.preventDefault();
            } else if (e.keyCode === 38) { // Up arrow
                currentFocus--;
                addActive(suggestions);
                e.preventDefault();
            } else if (e.keyCode === 13) { // Enter
                e.preventDefault();
                if (currentFocus > -1 && suggestions[currentFocus]) {
                    suggestions[currentFocus].click();
                } else if (this.value.trim()) {
                    searchBtn.click();
                }
            } else if (e.keyCode === 27) { // Escape
                suggestionsDiv.style.display = 'none';
            }
        });
        
        function addActive(suggestions) {
            if (!suggestions || suggestions.length === 0) return false;
            removeActive(suggestions);
            
            if (currentFocus >= suggestions.length) currentFocus = 0;
            if (currentFocus < 0) currentFocus = (suggestions.length - 1);
            
            suggestions[currentFocus].classList.add('active');
            suggestions[currentFocus].scrollIntoView({ block: 'nearest' });
        }
        
        function removeActive(suggestions) {
            for (let i = 0; i < suggestions.length; i++) {
                suggestions[i].classList.remove('active');
            }
        }
        
        function selectPlace(place) {
            searchInput.value = place.name + ', ' + place.region;
            suggestionsDiv.style.display = 'none';
            
            // Update location on map
            updateLocation(place.lat, place.lon, place.name + ', ' + place.region);
            
            if (map) {
                map.setView([place.lat, place.lon], 14);
            }
        }
        
        // Close suggestions when clicking outside
        document.addEventListener('click', function(e) {
            if (e.target !== searchInput && !suggestionsDiv.contains(e.target)) {
                suggestionsDiv.style.display = 'none';
            }
        });
    }
    
    // Search button click event
    if (searchBtn) {
        searchBtn.addEventListener('click', function() {
            const query = searchInput.value;
            searchLocation(query);
        });
    }
    
    // Search on Enter key
    if (searchInput && !suggestionsDiv) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchBtn.click();
            }
        });
    }

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

// ============================================
// Auto-load all cached locations from API
// ============================================
let allLocationMarkers = [];
let markerClusterGroup;

async function loadAllLocations() {
    try {
        console.log('Loading all cached locations...');
        
        const response = await fetch('/api/locations');
        
        if (!response.ok) {
            console.error('Failed to load locations');
            return;
        }
        
        const data = await response.json();
        
        if (data.status === 'processing') {
            console.log('Locations are being processed. Will retry in 10 seconds...');
            document.getElementById('mapInfo').innerHTML = '⏳ Processing locations... Please wait.';
            setTimeout(loadAllLocations, 10000); // Retry in 10 seconds
            return;
        }
        
        if (data.locations && data.locations.length > 0) {
            console.log(`Loaded ${data.locations.length} locations`);
            displayLocationsOnMap(data.locations);
            
            // Update info banner
            document.getElementById('mapInfo').innerHTML = `
                📍 <strong>${data.locations.length} locations loaded</strong> 
                <span style="font-size: 0.85em;">(Last updated: ${new Date(data.last_updated).toLocaleTimeString()})</span>
            `;
        } else {
            console.log('No locations available yet');
            document.getElementById('mapInfo').innerHTML = '⏳ No locations cached yet. Processing...';
        }
        
    } catch (error) {
        console.error('Error loading locations:', error);
    }
}

function displayLocationsOnMap(locations) {
    // Clear existing location markers (but keep user's selected marker)
    allLocationMarkers.forEach(m => {
        if (m !== marker) {
            map.removeLayer(m);
        }
    });
    allLocationMarkers = [];
    
    // Create custom icons based on score
    const getMarkerColor = (score) => {
        if (score >= 80) return '#22c55e'; // Green - Excellent
        if (score >= 60) return '#f59e0b'; // Amber - Good
        if (score >= 40) return '#ef4444'; // Red - Moderate
        return '#991b1b'; // Dark Red - Poor
    };
    
    const createCustomIcon = (score) => {
        const color = getMarkerColor(score);
        return L.divIcon({
            className: 'custom-marker',
            html: `<div style="
                background-color: ${color};
                width: 24px;
                height: 24px;
                border-radius: 50%;
                border: 2px solid white;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 10px;
            ">${Math.round(score)}</div>`,
            iconSize: [24, 24],
            iconAnchor: [12, 12]
        });
    };
    
    // Add markers for each location
    locations.forEach(location => {
        const locationMarker = L.marker(
            [location.latitude, location.longitude],
            { icon: createCustomIcon(location.eat_score_now) }
        ).addTo(map);
        
        // Create detailed popup
        const popupContent = `
            <div style="min-width: 200px;">
                <h3 style="margin: 0 0 8px 0; color: var(--accent);">
                    SafeBite Score: ${location.eat_score_now.toFixed(1)}
                </h3>
                <p style="margin: 4px 0;"><strong>Location:</strong> ${location.latitude.toFixed(5)}, ${location.longitude.toFixed(5)}</p>
                <p style="margin: 4px 0;"><strong>PM2.5:</strong> ${location.external_pm25.toFixed(1)} μg/m³</p>
                <p style="margin: 4px 0;"><strong>Risk Level:</strong> <span style="color: ${getMarkerColor(location.eat_score_now)};">${location.risk_level_now}</span></p>
                <p style="margin: 4px 0;"><strong>Status:</strong> ${location.current_anomaly}</p>
                <p style="margin: 8px 0 4px 0; font-size: 0.85em; color: var(--muted);">
                    30-min forecast: ${location.eat_score_future.toFixed(1)}
                </p>
                <button onclick="selectLocation(${location.latitude}, ${location.longitude})" 
                        style="margin-top: 8px; padding: 6px 12px; background: var(--accent); color: white; border: none; border-radius: 6px; cursor: pointer; width: 100%;">
                    Select This Location
                </button>
            </div>
        `;
        
        locationMarker.bindPopup(popupContent);
        
        // Click to select this location
        locationMarker.on('click', function() {
            updateCoordinates(location.latitude, location.longitude);
        });
        
        allLocationMarkers.push(locationMarker);
    });
    
    // Fit map to show all markers
    if (allLocationMarkers.length > 0) {
        const group = new L.featureGroup(allLocationMarkers);
        map.fitBounds(group.getBounds().pad(0.1));
    }
}

// Function to select a location from popup button
function selectLocation(lat, lon) {
    updateLocation(lat, lon);
    // Scroll to analyze button
    document.getElementById('submit').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Make function globally available
window.selectLocation = selectLocation;

// Auto-refresh locations every 5 minutes
setInterval(loadAllLocations, 5 * 60 * 1000);

// Load locations when map is ready
if (typeof map !== 'undefined' && map) {
    setTimeout(loadAllLocations, 2000); // Wait 2 seconds for map to initialize
} else {
    // Wait for map initialization
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(loadAllLocations, 3000);
    });
}