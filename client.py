import requests
body = {
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.3254
    }
response = requests.post(url = 'http://127.0.0.1:8000/score',
              json = body)
print (response.json())

