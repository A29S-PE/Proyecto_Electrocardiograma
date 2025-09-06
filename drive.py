from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import json

FOLDER_ID = "1i_L7rPHuCHDt2heyeeKhTJA7wdz9-LOQ"
INDEX_FILE_ID = "1cfG6JUdAZ3SknBu-SG2shsiB5cvdxQAK"

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

credentials = service_account.Credentials.from_service_account_file('substantial-art-471303-h3-1af3638890c8.json', scopes=SCOPES)

drive_service = build('drive', 'v3', credentials=credentials)

request = drive_service.files().get_media(fileId=INDEX_FILE_ID)
fh = io.BytesIO()
downloader = MediaIoBaseDownload(fh, request)
done = False
while not done:
    _, done = downloader.next_chunk()
fh.seek(0)
records_index = json.loads(fh.read().decode("utf-8"))
print(f"Índice cargado: {len(records_index)} registros")


def get_records_index():
    return records_index


def find_file_id(record_id: str, ext: str):
    filename = f"{record_id}.{ext}"
    query = f"name = '{filename}' and trashed = false"
    results = drive_service.files().list(
        q=query,
        spaces='drive',
        fields="files(id, name)",
        pageSize=1
    ).execute()
    files = results.get('files', [])
    if files:
        return files[0]['id']
    return None