from flask import Flask, render_template, request
import os
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
from collections import Counter
from pydub import AudioSegment
import joblib  
import librosa
import numpy as np
import pdfkit
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

model_path = r'random_forest_emotion_polarity_model.pkl'
emotion_model = joblib.load(model_path)

emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
polarity_scores = {
    'Angry': -0.8,
    'Sad': -0.5,
    'Neutral': 0.0,
    'Happy': 0.8,
    'Surprise': 0.5
}

app = Flask(__name__)

os.environ["API_KEY"] = "AIzaSyBoakTUf4je7Et5z8iGGwmza4nniyekXBY"

genai.configure(api_key=os.environ["API_KEY"])

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio_segment = AudioSegment.from_mp3(mp3_file_path)
    audio_segment.export(wav_file_path, format="wav")

def extract_features(file_path, sr=22050, n_mfcc=13):
    audio, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def predict_emotion(file_path):
    mfcc = extract_features(file_path).reshape(1, -1)
    predicted_emotion = emotion_model.predict(mfcc)[0]
    polarity_score = polarity_scores.get(predicted_emotion, 0.0)
    return predicted_emotion, polarity_score

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Error with the service: {e}"

def generate_feedback(transcription):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Provide feedback on the following interview: {transcription}")
    return response.text if response else "No response received from the API."

def generate_personalized_feedback(transcription, query):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"{query}: {transcription}")
    return response.text if response else "No response received from the API."

def generate_pie_chart(transcription):
    stop_words = set(STOPWORDS)
    words = transcription.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    stop_word_count = len(words) - len(filtered_words)

    analysis_data = {
        "Stop Words": stop_word_count,
        "Filtered Words": len(filtered_words),
        "Clarity": len(filtered_words) / len(words),
        "Listenability": 1 - (stop_word_count / len(words)),
        "Pace": len(words) / len(transcription)
    }

    colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']
    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(
        analysis_data.values(),
        labels=analysis_data.keys(),
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.3)
    )
    legend_labels = [f'{label}: {value:.2f}' for label, value in zip(analysis_data.keys(), analysis_data.values())]
    plt.legend(wedges, legend_labels, title="Analysis", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.savefig('static/pie_chart.png', bbox_inches='tight')
    plt.close()

def feedback_to_audio(feedback_text, output_audio_file):
    cleaned_text = feedback_text.replace("*", "")
    tts = gTTS(text=cleaned_text, lang='en')
    tts.save(output_audio_file)

def send_email_with_pie_chart(subject, body, to_email, pie_chart_image):
    from_email = "swethajayaraman99@gmail.com"
    from_password = "tnxbtbdelntadkfn"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    with open(pie_chart_image, "rb") as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename= {pie_chart_image}")
        msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, from_password)
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()
    print(f"Email sent to: {to_email}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        recipient_email = request.form.get('email')
        query = request.form.get('query')

        if file:
            audio_path_1 = 'tester.mp3'
            audio_path = 'tester.wav'
            
            file.save(audio_path)
            #convert_mp3_to_wav(audio_path_1, audio_path)

            transcription = transcribe_audio(audio_path)
            feedback = generate_feedback(transcription)
            personalized_feedback = generate_personalized_feedback(transcription, query) if query else "No query provided."
            generate_pie_chart(transcription)

            feedback_audio_path = 'static/feedback_audio.mp3'
            feedback_to_audio(feedback, feedback_audio_path)

            predicted_emotion, polarity_score = predict_emotion(audio_path)

            email_subject = "Your Speech Evaluation Results"
            email_body = (
                f"Transcription: {transcription}\n\n"
                f"Feedback: {feedback}\n\n"
                f"Personalized Feedback: {personalized_feedback}\n\n"
                f"Predicted Emotion: {predicted_emotion}\n\n"
                f"Polarity Score: {polarity_score}\n\n"
                "Please find attached the pie chart of the speech analysis."
            )
            pie_chart_image = 'static/pie_chart.png'

            send_email_with_pie_chart(
                subject=email_subject,
                body=email_body,
                to_email=recipient_email,
                pie_chart_image=pie_chart_image
            )

            return render_template('index.html', transcription=transcription, feedback_audio=feedback_audio_path,
                                   personalized_feedback=personalized_feedback,
                                   predicted_emotion=predicted_emotion, polarity_score=polarity_score)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
