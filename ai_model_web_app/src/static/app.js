document.addEventListener('DOMContentLoaded', function() {
    const submitBtn = document.getElementById('submit');
    const latitudeInput = document.getElementById('latitude');
    const longitudeInput = document.getElementById('longitude');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');

    submitBtn.addEventListener('click', async function() {
        const latitude = parseFloat(latitudeInput.value);
        const longitude = parseFloat(longitudeInput.value);

        // Validation
        if (isNaN(latitude) || isNaN(longitude)) {
            showError('Please enter valid latitude and longitude values');
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

    // Allow Enter key to submit
    [latitudeInput, longitudeInput].forEach(input => {
        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                submitBtn.click();
            }
        });
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