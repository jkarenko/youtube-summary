import yt_dlp
import npyscreen
import os
import threading
import openai
from whispercpp import Whisper
from pathlib import Path

w = Whisper('tiny')

OUTPUT_DIR = './transcribe-audio/'
openai.api_key = os.getenv('OPENAI_API_KEY')


def summarize_text(text):
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=text,
        temperature=0.3,
        max_tokens=100
    )
    return response.choices[0].text.strip()


def download_audio_from_youtube(url):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'format': 'bestaudio/best',
        'outtmpl': './transcribe-audio/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '16',
        }],
        'postprocessor_args': [
            '-ar', '16000', '-ac', '1', '-sample_fmt', 's16'
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info_dict)
        filename = Path(filename).with_suffix('.wav').name

    return filename or None


def download_video_from_youtube(url):
    ydl_opts = {
        'format': 'best',
        'outtmpl': f'{OUTPUT_DIR}%(title)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def speech_to_text(audio_file):
    transcription = w.transcribe(audio_file)
    return w.extract_text(transcription)


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


class App(npyscreen.NPSAppManaged):
    def onStart(self):
        self.addForm('MAIN', MainForm, name="YouTube Transcriber")


class MainForm(npyscreen.ActionFormMinimal):
    def transcribe_audio(self, files):
        transcription = ''.join(speech_to_text(OUTPUT_DIR + files['audio']))
        minutes = meeting_minutes(transcription=transcription)
        print(minutes)
        files['transcription'] = f'{files["audio"]}-transcription.txt'
        files['minutes'] = f'{files["audio"]}-minutes.txt'
        with open(OUTPUT_DIR + f'{files["transcription"]}', 'w') as f:
            f.write(transcription)
        with open(OUTPUT_DIR + f'{files["minutes"]}', 'w') as f:
            for title, text in minutes.items():
                title = title.replace('_', ' ').strip().title()
                f.write(f"{title}\n\n{text}\n\n\n")

    def create(self):
        self.url = self.add(npyscreen.TitleText,
                            name="Enter the YouTube URL: ")
        self.choice = self.add(npyscreen.TitleSelectOne, max_height=4, value=[0], name="Select operation:",
                               values=["Download audio", "Download video", "Transcribe Audio"], scroll_exit=True)

    def on_ok(self):
        url = self.url.value
        choice = self.choice.value[0]

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_info = ydl.extract_info(url, download=False)

        info_string = f"\nTitle: {video_info['title']}\n\nDescription: {video_info['description']}\n"
        npyscreen.notify_confirm(info_string, title="Video Info")

        self.parentApp.switchForm(None)

        files = {}
        if choice in [0, 2]:
            files['audio'] = download_audio_from_youtube(url)
        if choice == 1:
            files['video'] = download_video_from_youtube(url)
        if choice == 2:
            transcription_thread = threading.Thread(
                target=self.transcribe_audio, args=(files,))
            transcription_thread.start()


if __name__ == '__main__':
    App = App()
    App.run()
