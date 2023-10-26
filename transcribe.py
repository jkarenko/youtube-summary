import yt_dlp
import os
import threading
import openai
import sys
from pathlib import Path
from pydub import AudioSegment

OUTPUT_DIR = './transcribe-audio/'
openai.api_key = os.getenv('OPENAI_API_KEY')
file_extension = 'mp3'


def download_audio_from_youtube(url):
    print("processing audio...")
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'format': 'bestaudio/best',
        'outtmpl': './transcribe-audio/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': file_extension,
            'preferredquality': '32',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info_dict)
        filename = Path(filename).with_suffix('.' + file_extension).name

    return filename or None


def speech_to_text(audio_file):
    print("transcribing audio...")
    audio = open(audio_file, "rb")
    response = openai.Audio.transcribe('whisper-1', audio)
    return response['text']


def meeting_minutes(transcription):
    abstract_summary = abstract_summary_extraction(transcription)
    key_points = key_points_extraction(transcription)
    action_items = action_item_extraction(transcription)
    sentiment = sentiment_analysis(transcription)
    return {
        'abstract_summary': abstract_summary,
        'key_points': key_points,
        'action_items': action_items,
        'sentiment': sentiment
    }


def abstract_summary_extraction(transcription):
    print("creating abstract...")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


def key_points_extraction(transcription):
    print("extracting key points...")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in distilling information into key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


def sentiment_analysis(transcription):
    print("performing sentiment analysis...")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "As an AI with expertise in language and emotion analysis, your task is to analyze the sentiment of the following text. Please consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used. Indicate whether the sentiment is generally positive, negative, or neutral, and provide brief explanations for your analysis where possible."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


def action_item_extraction(transcription):
    print("extracting action items...")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an AI expert in analyzing conversations and extracting action items. Please review the text and identify any tasks, assignments, or actions that were agreed upon or mentioned as needing to be done. These could be tasks assigned to specific individuals, or general actions that the group has decided to take. Please list these action items clearly and concisely."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


def transcribe_audio(files):
    transcription = speech_to_text(OUTPUT_DIR + files['audio'])
    minutes = meeting_minutes(transcription=transcription)

    files['transcription'] = f'{files["audio"]}-transcription.txt'
    files['minutes'] = f'{files["audio"]}-minutes.md'
    with open(OUTPUT_DIR + f'{files["transcription"]}', 'w') as f:
        f.write(transcription)
    with open(OUTPUT_DIR + f'{files["minutes"]}', 'w') as f:
        f.write(f"# {files['audio']}\n")
        f.write(f"[{url}]({url})\n\n")
        for title, text in minutes.items():
            title = title.replace('_', ' ').strip(
            ).title().split('.' + file_extension)[0]
            f.write(f"## {title}\n\n{text}\n\n\n")

    print(f"All done! Find your files in {OUTPUT_DIR}")

def main(url):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        video_info = ydl.extract_info(url, download=False)

    info_string = f"\nTitle: {video_info['title']}\n\nDescription: {video_info['description']}\n"
    print(info_string)

    files = {}
    files['audio'] = download_audio_from_youtube(url)
    transcribe_audio(files)


if __name__ == '__main__':
    url = sys.argv[1]
    main(url)
