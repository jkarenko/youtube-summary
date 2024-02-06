import re
import string
import yt_dlp
import os
import openai
import sys
import subprocess
import tiktoken
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from pydub import AudioSegment

OUTPUT_DIR = './transcribe-audio/'
openai.api_key = os.getenv('OPENAI_API_KEY')
file_extension = 'mp3'
lemmatizer = WordNetLemmatizer()
MODEL = "gpt-4-turbo-preview"
MODEL_SRT = "gpt-3.5-turbo"
MIN_OUTPUT_TOKENS = 300
SAMPLE_RATE = '8000'


MODELS_INFO = {
    'gpt-3.5-turbo': {'max_tokens': 4097, 'per_1k_tokens_input': 0.0015, 'per_1k_tokens_output': 0.002},
    'gpt-3.5-turbo-16k': {'max_tokens': 16385, 'per_1k_tokens_input': 0.003, 'per_1k_tokens_output': 0.004},
    'gpt-4': {'max_tokens': 8192, 'per_1k_tokens_input': 0.03, 'per_1k_tokens_output': 0.06},
    'gpt-4-32k': {'max_tokens': 32768, 'per_1k_tokens_input': 0.06, 'per_1k_tokens_output': 0.12},
    'gpt-4-turbo-preview': {'max_tokens': 128000, 'per_1k_tokens_input': 0.01, 'per_1k_tokens_output': 0.03},
    'whisper': {'max_size_mb': 25, 'cost_per_minute': 0.006},
}


def speed_up_audio(filename):
    print("Speeding up audio...")
    output_filename = Path(filename).with_suffix('.' + file_extension).name
    print(f"Creating {OUTPUT_DIR + output_filename}")
    subprocess.run(['ffmpeg', '-n', '-i', filename, '-filter:a', 'atempo=2.0', '-ar', SAMPLE_RATE, '-vn', '-ac', '1', '-q:a', '9', OUTPUT_DIR + output_filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    output_path = simulate_gsm_compression(OUTPUT_DIR + output_filename)
    return Path(output_path).name

def simulate_gsm_compression(input_filename, output_dir=OUTPUT_DIR):
    print("Simulating GSM compression...")
    audio = AudioSegment.from_file(input_filename)
    audio = audio.set_frame_rate(8000).set_sample_width(1)  # Set frame rate to 8000Hz and sample width to 1 byte
    output_filename = Path(input_filename).stem + ".mp3"
    output_path = os.path.join(output_dir, output_filename)
    audio.export(output_path, format="mp3", bitrate="13k")
    return output_path


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
    # audio_duration_minutes = len(audio.read()) / SAMPLE_RATE / 60
    audio_duration_minutes = AudioSegment.from_file(audio_file).duration_seconds / 60
    # if audio_duration_minutes > 30:
    #     print(f"Audio duration is {audio_duration_minutes} minutes, splitting into 30 minute chunks...")
    #     audio_chunks = AudioSegment.from_file(audio_file).split_to_mono()
    #     audio_duration_minutes = 0
    #     for chunk in audio_chunks:
    #         audio_duration_minutes += chunk.duration_seconds / 60
    #         if audio_duration_minutes >= 30:
    #             break
    #     audio = audio_chunks[0]
    #     audio_file = audio_file.split('.')[0] + '-chunk.mp3'
    #     audio.export(audio_file, format='mp3')
    #     print(f"New audio file is {audio_file}")
    #     print(f"New audio duration is {audio_duration_minutes} minutes")

    audio_cost = audio_duration_minutes * MODELS_INFO['whisper']['cost_per_minute']

    print(f"audio duration: {audio_duration_minutes} minutes")
    print(f"cost for audio to text: ${audio_cost}")
    print(f"transcribing audio from {audio_file}...")

    # response = openai.Audio.transcribe(model='whisper-1', file=audio, response_format='srt', language='en')
    response = openai.audio.transcriptions.create(model='whisper-1', file=audio, response_format='srt')

    print("\n")
    return response, audio_cost


def meeting_minutes(transcription, transcription_srt):
    system_texts = {
        "abstract": {"summary_type": "abstract", "system": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points."},
        # "key_points": {"summary_type": "key points", "system": "You are a proficient AI with a specialty in distilling information into key points or central claims. Based on the following text, identify and list the main points or claims that were discussed or brought up. These should be the most important ideas, findings, claims or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about. Output a numbered list of key points (e.g. 1. point 1)"},
        "key_points": {"summary_type": "key points", "system": "You are a proficient AI with a specialty in distilling information into central claims. Based on the following text, identify and list the main claims brought up. These should be the most important claims that are crucial to the essence of the argument. Your goal is to provide a list that someone could read to quickly understand what was claimed. Output a numbered list of the claims (e.g. 1. point 1)"},
        "truthfulness": {"summary_type": "claim evaluation", "system": "Examine the text and categorize it as TRUE, FALSE, SUBJECTIVE, or UNKNOWN—respond with classification - reason. Assign 'FALSE' with verifiable incorrect information and provide a brief correction. Use 'SUBJECTIVE' for opinions or interpretive statements, with a succinct explanation. If a statement cannot be verified with the training data, classify it as UNKNOWN. Ignore minor typos in names."},
        "action_items": {"summary_type": "action items", "system": "You are an AI expert in analyzing conversations and extracting action items. Please review the text and identify any tasks, assignments, or actions that were agreed upon or mentioned as needing to be done. These could be tasks assigned to specific individuals, or general actions that the group has decided to take. Please list these action items clearly and concisely."},
        "sentiment_analysis": {"summary_type": "sentiment analysis", "system": "As an AI with expertise in language and emotion analysis, your task is to analyze the sentiment of the following text. Please consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used. Indicate whether the sentiment is generally positive, negative, or neutral, and provide brief explanations for your analysis where possible."},
    }

    abstract_summary = summary_extraction(transcription, system_texts['abstract']['summary_type'], system_texts['abstract']['system'])
    # key_points = summary_extraction(transcription_srt, system_texts['key_points']['summary_type'], system_texts['key_points']['system'], model=MODEL_SRT)
    key_points = summary_extraction(transcription, system_texts['key_points']['summary_type'], system_texts['key_points']['system'], model='gpt-4-turbo-preview')
    claims = []
    for point in key_points['content'].split('\n'):
        if re.match(r"^\d+\.", point):
            claims += point + "\n\n\t" + summary_extraction(f"text:\n{transcription}\n\nclaim:\n{point}", system_texts['truthfulness']['summary_type'], system_texts['truthfulness']['system'], model='gpt-4-turbo-preview')['content'] + "\n\n"
    # truthfulness = summary_extraction(key_points['content'], system_texts['truthfulness']['summary_type'], system_texts['truthfulness']['system'], model='gpt-4')
    action_items = summary_extraction(transcription, system_texts['action_items']['summary_type'], system_texts['action_items']['system'])
    sentiment = summary_extraction(transcription, system_texts['sentiment_analysis']['summary_type'], system_texts['sentiment_analysis']['system'])

    # kp = [kp for kp in key_points['content'].split('\n') if re.match(r"^\d+\.", kp)]
    # tr = ['. '.join(tr.split('. ')[1:]) for tr in truthfulness['content'].split('\n') if re.match(r"^\d+\.", tr)]

    # combined_kp_tr = []
    # for i in range(len(kp)):
    #     combined_kp_tr.append(f"{kp[i]}\n\n\t{tr[i]}\n\n")


    return {
        'abstract_summary'  : abstract_summary['content'],
        # 'key_points'        : "".join(combined_kp_tr),
        # 'central_claims'    : "".join(combined_kp_tr),
        'central_claims'    : "".join(claims),
        # 'truthfulness'      : truthfulness['content'],
        'action_items'      : action_items['content'],
        'sentiment'         : sentiment['content'],
    }, {
        'abstract_summary'  : (abstract_summary['prompt_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_input'] + abstract_summary['completion_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_output']) / 1000,
        'key_points'        : (key_points['prompt_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_input'] + key_points['completion_tokens'] * MODELS_INFO[MODEL_SRT]['per_1k_tokens_output']) / 1000,
        # 'truthfulness'      : (truthfulness['prompt_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_input'] + truthfulness['completion_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_output']) / 1000,
        'truthfulness'      : 0,
        'action_items'      : (action_items['prompt_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_input'] + action_items['completion_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_output']) / 1000,
        'sentiment'         : (sentiment['prompt_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_input'] + sentiment['completion_tokens'] * MODELS_INFO[MODEL]['per_1k_tokens_output']) / 1000,
    }


def summary_extraction(transcription, summary_type, system_message=None, model=MODEL):
    print(f"creating {summary_type}...")
    response = openai.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )

    return {
        'content': response.choices[0].message.content,
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens
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


def count_openai_tokens(transcription, model=MODEL):
    enc = tiktoken.encoding_for_model(model_name=model)
    return len(enc.encode(text=transcription))


def srt_to_text(transcription_srt):
    print("converting srt to text...")

    lines = []

    for line in transcription_srt.splitlines():
        if re.match(r'\d{2}:\d{2}:\d{2},\d{3}', line):
            continue
        if line.strip() == '':
            continue
        if re.match(r'^\d+$', line):
            continue
        lines.append(line)

    return "\n".join(lines)


def transcribe_audio(files):
    global MODEL, MODEL_SRT
    transcription_srt = None
    transcription = None
    minutes = None

    costs = {
        'abstract_summary':     0,
        'key_points':           0,
        'truthfulness':         0,
        'action_items':         0,
        'sentiment':            0,
    }
    audio_cost = 0

    # check if transcription file exists
    if os.path.exists(OUTPUT_DIR + files['audio'] + '-transcription.txt'):
        print("Transcription file already exists, skipping...")
        transcription = open(file=OUTPUT_DIR + files['audio'] + '-transcription.txt', mode='r').read()
    if os.path.exists(OUTPUT_DIR + files['audio'] + '-transcription.srt'):
        print("Transcription SRT file already exists, skipping...")
        transcription_srt = open(file=OUTPUT_DIR + files['audio'] + '-transcription.txt', mode='r').read()
    else:
        transcription_srt, audio_cost = speech_to_text(audio_file=OUTPUT_DIR + files['audio'])
        files['transcription_srt'] = f'{files["audio"]}-transcription.srt'

        transcription = srt_to_text(transcription_srt)
        files['transcription'] = f'{files["audio"]}-transcription.txt'

        with open(OUTPUT_DIR + f'{files["transcription_srt"]}', 'w') as f:
            f.write(transcription_srt)

        with open(OUTPUT_DIR + f'{files["transcription"]}', 'w') as f:
            f.write(transcription)


    max_tokens = MODELS_INFO[MODEL]['max_tokens']
    print(f"\nMax tokens for {MODEL}: {max_tokens}")

    tokens = count_openai_tokens(transcription=transcription, model=MODEL)
    tokens_srt = count_openai_tokens(transcription=transcription_srt, model=MODEL_SRT)

    print(f"Transcript tokens required: {tokens}\n")

    minimized_text = minimize_text(transcription=transcription)
    minimized_tokens = count_openai_tokens(transcription=minimized_text, model=MODEL)

    print(f"Minimized transcript tokens: {minimized_tokens}")
    print(f"Transcript SRT tokens required: {tokens_srt}\n")
    print("Using SRT transcript for key points and minimized text for everything else...\n")

    if minimized_tokens > max_tokens - MIN_OUTPUT_TOKENS:
        # new_model = next((k for k, v in MAX_TOKENS.items() if v > minimized_tokens + 100 and k.startswith('-'.join(MODEL.split('-')[:2]))), None)
        new_model = next((k for k, v in MODELS_INFO.items() if v['max_tokens'] > minimized_tokens + MIN_OUTPUT_TOKENS), None)
        # transcription = truncate_text(transcription=transcription)
        print(
            f"Amount of input tokens ({minimized_tokens}) would only leave {max_tokens - minimized_tokens} tokens for output. Minimum output tokens is {MIN_OUTPUT_TOKENS} and {MODEL} has a max token limit of {max_tokens}.")
        print(
            f"Continuing with {new_model}, which has a max token limit of {MODELS_INFO[new_model]['max_tokens']}")
        print(f"This might increase your costs significantly.")
        print(f"Current cost for input only: ${MODELS_INFO[MODEL]['per_1k_tokens_input'] * 4 * minimized_tokens / 1000}")
        print(f"New cost for input only: ${MODELS_INFO[new_model]['per_1k_tokens_input'] * 4 * minimized_tokens / 1000}")

        MODEL = new_model

    if tokens_srt > max_tokens - MIN_OUTPUT_TOKENS:
        # new_model = next((k for k, v in MAX_TOKENS.items() if v > minimized_tokens + 100 and k.startswith('-'.join(MODEL.split('-')[:2]))), None)
        new_model_srt = next((k for k, v in MODELS_INFO.items() if v['max_tokens'] > tokens_srt + MIN_OUTPUT_TOKENS), None)
        # transcription = truncate_text(transcription=transcription)
        print(
            f"Amount of input tokens ({tokens_srt}) would only leave {max_tokens - tokens_srt} tokens for output. Minimum output tokens is {MIN_OUTPUT_TOKENS} and {MODEL} has a max token limit of {max_tokens}.")
        print(
            f"Continuing with {new_model_srt}, which has a max token limit of {MODELS_INFO[new_model_srt]['max_tokens']}")
        print(f"This might increase your costs significantly.")
        print(f"Current cost for input only: ${MODELS_INFO[MODEL]['per_1k_tokens_input'] * 4 * tokens_srt / 1000}")
        print(f"New cost for input only: ${MODELS_INFO[new_model_srt]['per_1k_tokens_input'] * 4 * tokens_srt / 1000}")

        MODEL_SRT = new_model_srt

    if os.path.exists(path=OUTPUT_DIR + files['audio'] + '-minutes.md'):
        print("Minutes file already exists, skipping minutes...")
        minutes = open(
            file=OUTPUT_DIR +
            files['audio'] + '-minutes.md', mode='r'
        ).read()
    else:
        minutes, costs = meeting_minutes(transcription=minimized_text, transcription_srt=transcription_srt)
        files['minutes'] = f'{files["audio"]}-minutes.md'
        with open(OUTPUT_DIR + f'{files["minutes"]}', 'w') as f:
            f.write(f"# {files['audio']}\n")
            f.write(f"[{url}]({url})\n\n")
            for title, text in minutes.items():
                title = title.replace('_', ' ').strip(
                ).title().split('.' + file_extension)[0]
                f.write(f"## {title}\n\n{text}\n\n\n")

    print(f"All done! Find your files in {OUTPUT_DIR}")
    total_cost = costs['abstract_summary'] + costs['key_points'] + costs['truthfulness'] + costs['action_items'] + costs['sentiment'] + audio_cost
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
        transcribe_audio(files=files)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception:
        raise Exception("An error occurred")


if __name__ == '__main__':
    url = sys.argv[1]
    main(url)
