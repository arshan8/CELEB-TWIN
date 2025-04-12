# Celebrity Twin Finder

Find out which celebrity you look like using face recognition!

## Setup

1. Install Docker and Docker Compose
2. Clone this repository
3. Create a `.env` file with:
   ```
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   ```

## Running the Application

1. Start Qdrant:
   ```bash
   docker-compose up -d
   ```

2. Generate celebrity embeddings (one-time setup):
   ```bash
   python src/scripts/generate_embeddings.py
   ```

3. Run the Streamlit app:
   ```bash
   cd src/app
   pip install -r requirements.txt
   streamlit run main.py
   ```

4. Open your browser and go to `http://localhost:8501`

## How It Works

1. The app uses FaceNet to detect faces and generate embeddings
2. Celebrity embeddings are stored in Qdrant vector database
3. When you take a photo, it's compared with celebrity embeddings
4. The closest match is shown as your celebrity twin

## Requirements

- Python 3.11+
- Docker
- Webcam (for taking photos)
