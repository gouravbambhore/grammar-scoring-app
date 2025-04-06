let recorder;
let audioChunks = [];

// Start Recording
document.getElementById("recordBtn").addEventListener("click", async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);

    recorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
    };

    recorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        const audioURL = URL.createObjectURL(audioBlob);
        const audioElement = document.getElementById("recordedAudio");

        // Set audio for playback
        audioElement.src = audioURL;
        audioElement.style.display = "block";

        // Enable "Generate Score" button
        document.getElementById("analyzeGrammarBtn").disabled = false;
    };

    recorder.start();
    document.getElementById("recordBtn").disabled = true;
    document.getElementById("stopBtn").disabled = false;
});

// Stop Recording
document.getElementById("stopBtn").addEventListener("click", () => {
    recorder.stop();
    document.getElementById("recordBtn").disabled = false;
    document.getElementById("stopBtn").disabled = true;
});

// Generate Score for Recorded Audio
document.getElementById("analyzeGrammarBtn").addEventListener("click", async () => {
    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
    const formData = new FormData();
    formData.append("audio", audioBlob, "recordedAudio.wav");

    const response = await fetch("http://127.0.0.1:5000/api/score", {
        method: "POST",
        body: formData,
    });

    const result = await response.json();
    document.getElementById("result").innerHTML = `
        <h2>Grammar Score: ${result.score} / 100</h2>
    `;
});

// Analyze Uploaded File
document.getElementById("uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData();
    const fileInput = document.getElementById("audioFile");
    formData.append("audio", fileInput.files[0]);

    const response = await fetch("http://127.0.0.1:5000/api/score", {
        method: "POST",
        body: formData,
    });

    const result = await response.json();
    document.getElementById("result").innerHTML = `
        <h2>Grammar Score: ${result.score} / 100</h2>
    `;
});
