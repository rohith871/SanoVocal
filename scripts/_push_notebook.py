import json
import requests
import sys

kg_token = sys.argv[1]
hf_token = sys.argv[2]

# Read notebook
with open(r'C:\Users\KOYA ROHITH\SanoVocal\notebooks\sanovocal_aya_qlora_training.ipynb', 'r', encoding='utf-8') as f:
    nb_content = f.read()

print(f'Notebook size: {len(nb_content)} bytes')

# Check if notebook already has HF_TOKEN injection
if 'HF_TOKEN' in nb_content:
    print('[OK] Notebook references HF_TOKEN')

# Push to Kaggle
headers = {
    'Authorization': f'Bearer {kg_token}',
    'Content-Type': 'application/json'
}

payload = {
    'id': 143625139,
    'newTitle': 'sanovocal-aya-qlora-training',
    'text': nb_content,
    'language': 'python',
    'kernelType': 'notebook',
    'isPrivate': True,
    'enableGpu': True,
    'enableInternet': True,
    'categoryIds': [],
    'datasetDataSources': [],
    'kernelDataSources': [],
    'competitionDataSources': []
}

resp = requests.post(
    'https://www.kaggle.com/api/v1/kernels/push',
    headers=headers,
    json=payload,
    timeout=60
)

print(f'HTTP Status: {resp.status_code}')
print(f'Response: {resp.text[:500]}')
