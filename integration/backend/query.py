import requests

ip = '3.225.204.208'
# ip = '127.0.0.1'
for port in [8000, 8001, 8002, 8003]:
    data = {'text':["我喜欢下雨。", "我讨厌他."]}
    url = f'http://{ip}:{port}/predict'
    try:
        response = requests.post(url, json=data)
    except:
        print(f'Port {port} not listening!')
        continue
    # response format is [{'label': 'positive', 'score': 0.98}, ..., {}]
    # TODO: response verification
    result = response.json()
    print(f'Port: {port}, API Response: {result}')