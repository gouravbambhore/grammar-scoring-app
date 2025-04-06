import os
import numpy as np
import pandas as pd
import librosa
import joblib
import ffmpeg
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Paths to model and CSV
MODEL_PATH = "grammar_scoring_model.joblib"
CSV_PATH = "final_submission.csv"
model = None
scores_df = None

def load_model():
    global model, scores_df
    try:
        print("Checking model path:", os.path.abspath(MODEL_PATH))
        print("Checking CSV path:", os.path.abspath(CSV_PATH))

        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully!")
        else:
            print(f"Error: Model file not found at {MODEL_PATH}")

        if os.path.exists(CSV_PATH):
            scores_df = pd.read_csv(CSV_PATH)
            print("CSV file loaded successfully!")
        else:
            print(f"Error: CSV file not found at {CSV_PATH}")
    except Exception as e:
        print(f"Error loading model or CSV: {e}")

def convert_to_wav(input_path, output_path):
    """
    Converts a .webm or other audio file to .wav format using FFmpeg.
    """
    try:
        print(f"Converting {input_path} to {output_path}...")

        # Ensure input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file {input_path} does not exist.")
            return None

        # Perform FFmpeg conversion
        ffmpeg.input(input_path).output(output_path).overwrite_output().run()

        # Verify output file creation
        if os.path.exists(output_path):
            print(f"Successfully converted {input_path} to {output_path}")
            return output_path
        else:
            print(f"Error: Output file {output_path} not created.")
            return None
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return None

def extract_features(audio_path, sr=22050, n_mfcc=12, duration=10):
    """
    Extracts audio features from a .wav file and creates a 50-dimensional feature vector.
    """
    try:
        print(f"Extracting features from: {audio_path}")
        audio_data, sr = librosa.load(audio_path, sr=sr, duration=duration)
        print(f"Audio Data Shape: {audio_data.shape}, Sample Rate: {sr}")

        # Extract features
        mfccs_mean = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc), axis=1)[:12]
        mfccs_std = np.std(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc), axis=1)[:12]
        chroma_mean = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sr), axis=1)[:8]
        spectral_contrast_mean = np.mean(librosa.feature.spectral_contrast(y=audio_data, sr=sr), axis=1)[:6]
        tonnetz_mean = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_data), sr=sr), axis=1)[:6]

        features = np.concatenate((mfccs_mean, mfccs_std, chroma_mean, spectral_contrast_mean, tonnetz_mean))

        # Ensure exactly 50 dimensions
        if len(features) < 50:
            features = np.pad(features, (0, 50 - len(features)), 'constant')
        elif len(features) > 50:
            features = features[:50]

        print(f"Features extracted: {features}")
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(50)

@app.route("/")
def home():
    """
    Home route for the API.
    """
    return (
        "<h1>Welcome to the Grammar Scoring API!</h1>"
        "<p>Use the <code>/api/score</code> endpoint to upload an audio file and get a grammar score.</p>"
        "<p>Check <code>/api/health</code> for health status of the API.</p>"
    )

@app.route("/api/score", methods=["POST"])
def score_audio():
    """
    Endpoint to process an audio file and return a grammar score.
    """
    temp_path = None
    temp_wav_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        print(f"Uploaded file: {audio_file.filename}")

        # Save the file temporarily
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.abspath(os.path.join(temp_dir, audio_file.filename))
        temp_wav_path = os.path.abspath(os.path.join(temp_dir, "converted_audio.wav"))  # Unique output file
        audio_file.save(temp_path)
        print(f"File saved at: {temp_path}")
        print(f"Does file exist? {os.path.exists(temp_path)}")

        # Convert to .wav format
        print(f"Input Path (absolute): {temp_path}")
        print(f"Output Path (absolute): {temp_wav_path}")
        converted_path = convert_to_wav(temp_path, temp_wav_path)
        if converted_path is None or not os.path.exists(converted_path):
            print(f"Error: Temporary WAV file {converted_path} was not created.")
            return jsonify({"error": "Audio conversion failed"}), 500

        # Extract features
        features = extract_features(converted_path)
        if not np.any(features):
            print("Failed to extract meaningful features.")
            return jsonify({"error": "Failed to extract features"}), 500

        # Predict score
        try:
            print("Making prediction...")
            score = float(model.predict([features])[0])
            print(f"Predicted Score: {score}")
            return jsonify({"score": score, "status": "success"})
        except ValueError as e:
            print(f"Model prediction error: {e}")
            return jsonify({"error": f"Model failed to predict score: {str(e)}"}), 500
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup temporary files
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Endpoint to check the health status of the API.
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "csv_loaded": scores_df is not None,
    })

if __name__ == "__main__":
    load_model()
    app.run(debug=False, host="0.0.0.0", port=5000)
