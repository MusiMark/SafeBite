import requests
import math

API_KEY = 'MYMAKDXGC9MNMRBZ'

#distance between the locations
def haversine_formula(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0

    # Convert coordinates from degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def get_pm2_5_value(site_id):
    SITE_ID = site_id
    BASE_URL = f'https://api.airqo.net/api/v2/devices/measurements/sites/{SITE_ID}/recent/?token={API_KEY}'

    try:
        # Send the GET request to the AirQo API
        response = requests.get(BASE_URL)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            return data['measurements'][0]['pm2_5']['value']

        else:
            print(f"Failed to retrieve data. Status Code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# get pm2.5 of nearest station
def nearest_pm2_5(mylat,mylon):
    BASE_URL = f'https://api.airqo.net/api/v2/devices/grids/summary/?token={API_KEY}'

    try:
        # Send the GET request to the AirQo API
        response = requests.get(BASE_URL)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

            uganda_grip = [2, 3, 4, 6, 7, 8, 9, 11]
            location_id = None
            site_distances = []

            for x in uganda_grip:
                for y in range(len(data['grids'][x]['sites'])):
                    latitude = data['grids'][x]['sites'][y]['approximate_latitude']
                    longitude = data['grids'][x]['sites'][y]['approximate_longitude']

                    distance = haversine_formula(mylat, mylon, latitude, longitude)

                    site_distances.append({
                        'id': data['grids'][x]['sites'][y]['_id'],
                        'distance': distance
                    })

            # Sort sites by distance
            sorted_sites = sorted(site_distances, key=lambda x: x['distance'])

            # Try the first nearest, then second if needed
            for site_info in sorted_sites:
                pm2_5_value = get_pm2_5_value(site_info['id'])
                if pm2_5_value is not None:
                    return pm2_5_value, site_info['distance'], site_info['id']

            return None , None, None

        else:
            print(f"Failed to retrieve data. Status Code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")


# if __name__ == '__main__':
def airqo_api(sample_lat, sample_lon):
    # sample_lat, sample_lon = 0.3520282097138751, 32.59490077052355

    # print(f"------The nearest pm_2.5 from ({sample_lat}, {sample_lon})------")
    value, distance, location_id = nearest_pm2_5(sample_lat, sample_lon)
    # print(f"PM2.5 is {value}")
    # print(f"Distance is {distance}")
    #
    # print(f"Nearest Location ID is {location_id}")

    return value, distance
