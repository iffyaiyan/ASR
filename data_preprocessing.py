import pandas as pd
import requests
import os
from pydub import AudioSegment
from google.oauth2 import service_account
from googleapiclient.discovery import build
import concurrent.futures

def download_file(url, output_path):
    response = requests.get(url)
    with open(output_path, 'wb') as f:
        f.write(response.content)

def segment_audio(input_path, output_dir, segment_length_ms=30000):
    audio = AudioSegment.from_mp3(input_path)
    segments = []
    for i, chunk in enumerate(audio[::segment_length_ms]):
        segment_path = f"{output_dir}/segment_{i}.wav"
        chunk.export(segment_path, format="wav")
        segments.append(segment_path)
    return segments

def process_sheet_data(sheet_id, range_name):
    creds = service_account.Credentials.from_service_account_file(
        'service_account.json',      # this path has to be given by the user, I don't have anything here to validate neither it was shared via mail, so I will give some name
        scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
    )
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=sheet_id, range=range_name).execute()
    data = result.get('values', [])
   
    df = pd.DataFrame(data[1:], columns=[
        "Sr. No.", "Case Name", "Case Number", "Hearing Date",
        "Transcript Link", "Oral Hearing Link", "Hearing Duration(in Minutes)",
        "mp3 format link"
    ])
    return df

def process_case(row, output_dir):
    case_dir = os.path.join(output_dir, f"case_{row['Sr. No.']}")
    os.makedirs(case_dir, exist_ok=True)
   
    audio_path = os.path.join(case_dir, "audio.mp3")
    download_file(row['mp3 format link'], audio_path)
    audio_segments = segment_audio(audio_path, case_dir)
   
    transcript_path = os.path.join(case_dir, "transcript.txt")
    download_file(row['Transcript Link'], transcript_path)
   
    return case_dir, audio_segments, transcript_path

def create_dataset(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
   
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_case = {executor.submit(process_case, row, output_dir): row for _, row in df.iterrows()}
        for future in concurrent.futures.as_completed(future_to_case):
            row = future_to_case[future]
            try:
                case_dir, audio_segments, transcript_path = future.result()
                print(f"Processed case {row['Sr. No.']}")
            except Exception as exc:
                print(f"Case {row['Sr. No.']} generated an exception: {exc}")

if __name__ == "__main__":
    sheet_id = "1fNy239xunVyCK3RZsVu6fWYgMi57vIfTLigT3YKgSWQ"
    range_name = "Sheet1!A1:H27501"
    output_dir = "court_transcription_dataset"
   
    df = process_sheet_data(sheet_id, range_name)
    create_dataset(df, output_dir)