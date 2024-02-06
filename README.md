# YouTube Summary

YouTube video to text to summary.

## Requirements

- [Download and install Docker](https://docs.docker.com/get-docker/)
- [Create a DockerHub account](https://hub.docker.com/signup)
- [OpenAI API key](https://platform.openai.com/account/api-keys)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/jkarenko/youtube-summary.git
```

1. Navigate to the project directory:

```bash
cd youtube-summary
```

1. Login to dockerhub with the account you created earlier:

```bash
docker login
```

1. Build the Docker image:

```bash
docker build --build-arg OPENAI_API_KEY=<your_key> -t youtube-summary .
```

## Usage

1. Run the Docker container:

```bash
docker run -it --rm -v ./transcribe-audio:/app/transcribe-audio youtube-summary <YouTube video URL>
```

1. The application will automatically start within the Docker container. Wait for the process to complete. Audio, transcript and minutes will appear in the `transcribe-audio` directory in the project root.
