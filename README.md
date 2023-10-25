# YouTube Summary

YouTube video to text to summary.

## Requirements

- [Docker](https://docs.docker.com/get-docker/)
- [OpenAI API key](https://platform.openai.com/account/api-keys)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/jkarenko/youtube-summary.git
   ```

2. Navigate to the project directory:

   ```bash
   cd youtube-summary
   ```

3. Build the Docker image:

   ```bash
   docker build --build-arg OPENAI_API_KEY=<your_key> -t youtube-summary .
   ```

## Usage

1. Run the Docker container:

    ```bash
    docker run -it --rm -v ./transcribe-audio:/app/transcribe-audio youtube-summary
    ```

2. The application will automatically start within the Docker container. Follow the prompts in the application to download and transcribe YouTube videos. Videos will appear in the `transcribe-audio` directory in the project root.
