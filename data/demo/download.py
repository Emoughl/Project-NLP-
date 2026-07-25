import requests

url = "https://vi.wikipedia.org/w/api.php"

params = {
    "action": "query",
    "format": "json",
    "titles": "Trí tuệ nhân tạo"
}

response = requests.get(url, params=params)

print(response.status_code)