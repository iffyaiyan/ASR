import os
import pandas as pd
from sklearn.model_selection import train_test_split

def align_audio_with_transcript(audio_segments, transcript_path, case_duration):
    with open(transcript_path, 'r') as f:
        full_transcript = f.read()
    
    words = full_transcript.split()
    words_per_segment = len(words) // len(audio_segments)
    
    aligned_data = []
    for i, segment in enumerate(audio_segments):
        start = i * words_per_segment
        end = start + words_per_segment if i < len(audio_segments) - 1 else None
        segment_transcript = " ".join(words[start:end])
        aligned_data.append({
            "audio_path": segment,
            "transcript": segment_transcript
        })
    
    return aligned_data

def create_common_voice_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    for case_dir in os.listdir(input_dir):
        case_path = os.path.join(input_dir, case_dir)
        if os.path.isdir(case_path):
            audio_segments = [f for f in os.listdir(case_path) if f.startswith("segment_") and f.endswith(".wav")]
            audio_segments.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
            transcript_path = os.path.join(case_path, "transcript.txt")
            
            case_number = case_dir.split("_")[1]
            case_row = df[df['Sr. No.'] == case_number].iloc[0]
            case_duration = int(case_row['Hearing Duration(in Minutes)'])
            
            aligned_data = align_audio_with_transcript(
                [os.path.join(case_dir, seg) for seg in audio_segments],
                transcript_path,
                case_duration
            )
            
            data.extend(aligned_data)
    
    df = pd.DataFrame(data)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_df.to_csv(os.path.join(output_dir, "train.tsv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(output_dir, "test.tsv"), sep="\t", index=False)
    
    with open(os.path.join(output_dir, "validated.tsv"), "w") as f:
        f.write("client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccent\tlocale\tsegment\n")

if __name__ == "__main__":
    input_dir = "court_transcription_dataset"
    output_dir = "common_voice_dataset"
    create_common_voice_dataset(input_dir, output_dir)