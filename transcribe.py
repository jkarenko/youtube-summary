import string
import yt_dlp
import os
import openai
import sys
import subprocess
import tiktoken
import time
import threading
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from pydub import AudioSegment

OUTPUT_DIR = './transcribe-audio/'
openai.api_key = os.getenv('OPENAI_API_KEY')
file_extension = 'mp3'
lemmatizer = WordNetLemmatizer()
MODEL = "gpt-3.5-turbo"
MIN_OUTPUT_TOKENS = 100
SAMPLE_RATE = 8000


class LoadingIndicator(threading.Thread):
    def __init__(self, lock):
        super(LoadingIndicator, self).__init__()
        self.stop_flag = threading.Event()
        self.lock = lock

    def run(self):
        while not self.stop_flag.is_set():
            for cursor in '|/-\\':
                with self.lock:
                    sys.stdout.write(cursor)
                    sys.stdout.flush()
                time.sleep(0.1)
                with self.lock:
                    sys.stdout.write('\b')

    def stop(self):
        self.stop_flag.set()


MODELS_INFO = {
    'gpt-3.5-turbo': {'max_tokens': 4097, 'per_1k_tokens_input': 0.0015, 'per_1k_tokens_output': 0.002},
    'gpt-3.5-turbo-16k': {'max_tokens': 16385, 'per_1k_tokens_input': 0.003, 'per_1k_tokens_output': 0.004},
    'gpt-4': {'max_tokens': 8192, 'per_1k_tokens_input': 0.03, 'per_1k_tokens_output': 0.06},
    'gpt-4-32k': {'max_tokens': 32768, 'per_1k_tokens_input': 0.06, 'per_1k_tokens_output': 0.12},
    'whisper': {'max_size_mb': 25, 'cost_per_minute': 0.006},
}


def loading_indicator():
    while True:
        for cursor in '|/-\\':
            sys.stdout.write(cursor)
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write('\b')


def speed_up_audio(filename):
    print("speeding up audio...")
    output_filename = Path(filename).with_suffix('.' + file_extension).name
    print(f"creating {output_filename}...")
    subprocess.run(['ffmpeg', '-n', '-i', filename, '-filter:a', 'atempo=2.0', '-ar', SAMPLE_RATE, '-vn', '-ac', '1', '-q:a', '9', OUTPUT_DIR + output_filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_filename


def download_audio_from_youtube(url, video_info):
    if os.path.exists(path=OUTPUT_DIR + video_info['title'] + '.mp3'):
        print("File already exists, skipping download...")
        return video_info['title'] + '.mp3'

    print("processing audio...")
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'format': 'bestaudio/best',
        'outtmpl': './transcribe-audio/%(title)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info_dict)
        # filename = Path(filename).with_suffix('.' + file_extension).name
    print(f"filename is {filename}")
    filename = speed_up_audio(filename)
    file_size = os.path.getsize(filename=OUTPUT_DIR + filename) / 1000000
    if file_size >= 25:
        print(f"File size is {file_size}MB, maximum file size is 25MB")
        exit(1)
    return filename or None


def speech_to_text(audio_file):
    print(f"opening audio file {audio_file}...")

    audio = open(audio_file, "rb")
    audio_duration_minutes = len(audio.read()) / SAMPLE_RATE / 60
    audio_cost = audio_duration_minutes * MODELS_INFO['whisper']['cost_per_minute']

    print(f"audio duration: {audio_duration_minutes} minutes")
    print(f"cost for audio to text: ${audio_cost}")
    print(f"transcribing audio from {audio_file}...")

    response = openai.Audio.transcribe(model='whisper-1', file=audio, response_format='srt', language='en')

    print("\n")
    return response, audio_cost


def meeting_minutes(transcription):
    abstract_summary = abstract_summary_extraction(transcription)
    key_points = key_points_extraction(transcription)
    action_items = action_item_extraction(transcription)
    sentiment = sentiment_analysis(transcription)
    return {
        'abstract_summary'  : abstract_summary['content'],
        'key_points'        : key_points['content'],
        'action_items'      : action_items['content'],
        'sentiment'         : sentiment['content'],
    }, {
        'abstract_summary'  : (abstract_summary['prompt_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_input'] + abstract_summary['completion_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_output']) / 1000,
        'key_points'        : (key_points['prompt_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_input'] + key_points['completion_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_output']) / 1000,
        'action_items'      : (action_items['prompt_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_input'] + action_items['completion_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_output']) / 1000,
        'sentiment'         : (sentiment['prompt_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_input'] + sentiment['completion_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_output']) / 1000,
    }


def abstract_summary_extraction(transcription):
    print("creating abstract...")
    response = openai.ChatCompletion.create(
        model=MODEL,
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

    return {
        'content': response['choices'][0]['message']['content'],
        'prompt_tokens': response['usage']['prompt_tokens'],
        'completion_tokens': response['usage']['completion_tokens']
    }


def key_points_extraction(transcription):
    print("extracting key points...")
    response = openai.ChatCompletion.create(
        model=MODEL,
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

    return {
        'content': response['choices'][0]['message']['content'],
        'prompt_tokens': response['usage']['prompt_tokens'],
        'completion_tokens': response['usage']['completion_tokens']
    }


def sentiment_analysis(transcription):
    print("performing sentiment analysis...")
    response = openai.ChatCompletion.create(
        model=MODEL,
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

    return {
        'content': response['choices'][0]['message']['content'],
        'prompt_tokens': response['usage']['prompt_tokens'],
        'completion_tokens': response['usage']['completion_tokens']
    }


def action_item_extraction(transcription):
    print("extracting action items...")
    response = openai.ChatCompletion.create(
        model=MODEL,
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

    return {
        'content': response['choices'][0]['message']['content'],
        'prompt_tokens': response['usage']['prompt_tokens'],
        'completion_tokens': response['usage']['completion_tokens']
    }


def minimize_text(transcription):
    stop_words = set(stopwords.words('english'))

    sentences = sent_tokenize(transcription)

    minimized_transcription = []
    for sent in sentences:
        words = word_tokenize(sent.lower())
        words = [lemmatizer.lemmatize(
            word) for word in words if word not in stop_words and word not in string.punctuation]
        minimized_transcription.append(' '.join(words))

    return ' '.join(minimized_transcription)


def count_openai_tokens(transcription):
    enc = tiktoken.encoding_for_model(model_name=MODEL)
    return len(enc.encode(text=transcription))


def truncate_text(transcription):
    max_tokens = MODELS_INFO[MODEL]['max_tokens'] - MIN_OUTPUT_TOKENS
    enc = tiktoken.encoding_for_model(model_name=MODEL)
    return enc.decode(enc.encode(text=transcription)[:max_tokens])


def transcribe_audio(files):
    global MODEL
    transcription = None
    minutes = None

    costs = {
        'abstract_summary':     0,
        'key_points':           0,
        'action_items':         0,
        'sentiment':            0,
    }
    audio_cost = 0

    # check if transcription file exists
    if os.path.exists(OUTPUT_DIR + files['audio'] + '-transcription.txt'):
        print("Transcription file already exists, skipping transcription...")
        transcription = open(
            file=OUTPUT_DIR + files['audio'] + '-transcription.txt', mode='r'
        ).read()
    else:
        transcription, audio_cost = speech_to_text(audio_file=OUTPUT_DIR + files['audio'])
        files['transcription'] = f'{files["audio"]}-transcription.txt'
        with open(OUTPUT_DIR + f'{files["transcription"]}', 'w') as f:
            f.write(transcription)

    max_tokens = MODELS_INFO[MODEL]['max_tokens']
    print(f"\nMax tokens for {MODEL}: {max_tokens}")
    tokens = count_openai_tokens(transcription=transcription)
    print(f"Raw transcript tokens: {tokens}\n")
    minimized_text = minimize_text(transcription=transcription)
    minimized_tokens = count_openai_tokens(transcription=minimized_text)
    print(f"minimized transcript tokens: {minimized_tokens}")
    print("Using minimized text for minutes...\n")
    if minimized_tokens > max_tokens - MIN_OUTPUT_TOKENS:
        # new_model = next((k for k, v in MAX_TOKENS.items() if v > minimized_tokens + 100 and k.startswith('-'.join(MODEL.split('-')[:2]))), None)
        new_model = next((k for k, v in MODELS_INFO.items() if v['max_tokens'] > minimized_tokens + 100), None)
        # transcription = truncate_text(transcription=transcription)
        print(
            f"Amount of input tokens ({minimized_tokens}) would only leave {max_tokens - minimized_tokens} tokens for output. Minimum output tokens is {MIN_OUTPUT_TOKENS} and {MODEL} has a max token limit of {max_tokens}.")
        print(
            f"Continuing with {new_model}, which has a max token limit of {MODELS_INFO[new_model]['max_tokens']}")
        print(f"This might increase your costs significantly.")
        print(f"Current cost for input only: ${MODELS_INFO[MODEL]['per_1k_tokens_input'] * 4 * minimized_tokens / 1000}")
        print(f"New cost for input only: ${MODELS_INFO[new_model]['per_1k_tokens_input'] * 4 * minimized_tokens / 1000}")
        MODEL = new_model
    # input("Press Enter to continue or Ctrl+C to exit...")

    # check if minutes file exists
    if os.path.exists(path=OUTPUT_DIR + files['audio'] + '-minutes.md'):
        print("Minutes file already exists, skipping minutes...")
        minutes = open(
            file=OUTPUT_DIR +
            files['audio'] + '-minutes.md', mode='r'
        ).read()
    else:
        minutes, costs = meeting_minutes(transcription=minimized_text)
        files['minutes'] = f'{files["audio"]}-minutes.md'
        with open(OUTPUT_DIR + f'{files["minutes"]}', 'w') as f:
            f.write(f"# {files['audio']}\n")
            f.write(f"[{url}]({url})\n\n")
            for title, text in minutes.items():
                title = title.replace('_', ' ').strip(
                ).title().split('.' + file_extension)[0]
                f.write(f"## {title}\n\n{text}\n\n\n")

    print(f"All done! Find your files in {OUTPUT_DIR}")
    total_cost = costs['abstract_summary'] + costs['key_points'] + costs['action_items'] + costs['sentiment'] + audio_cost
    print(f"Total cost: ${total_cost}")


def main(url):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        video_info = ydl.extract_info(url, download=False)

    video_info_text = f"\nTitle: {video_info['title']}\n\nDescription: {video_info['description']}\n"
    print(video_info_text)

    files = {}
    files['audio'] = download_audio_from_youtube(
        url=url, video_info=video_info)
    try:
        lock = threading.Lock()
        loading_indicator = LoadingIndicator(lock)
        loading_indicator.start()
        transcribe_audio(files=files)
        loading_indicator.stop()
    except KeyboardInterrupt:
        print("\nExiting...")
    except:
        raise Exception("An error occurred")
    finally:
        loading_indicator.stop()

if __name__ == '__main__':
    url = sys.argv[1]
    main(url)
