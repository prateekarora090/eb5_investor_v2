from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os
from dotenv import load_dotenv

load_dotenv('secrets/.env')

def get_drive_service():
    # Try service account first
    service_account_file = 'secrets/service_account_key.json'
    if os.path.exists(service_account_file):
        creds = service_account.Credentials.from_service_account_file(
            service_account_file, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
    return build('drive', 'v3', credentials=creds)

def list_files_in_folder(folder_id):
    service = get_drive_service()
    results = service.files().list(
        q=f"'{folder_id}' in parents",
        fields="nextPageToken, files(id, name, mimeType)"
    ).execute()
    items = results.get('files', [])
    return items

def read_file_from_drive(file_id):
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    return file.getvalue()