import cv2
import face_recognition
import os
import speech_recognition as sr
import pyttsx3

recognizer = sr.Recognizer()

engine = pyttsx3.init()

def recognize_speech():
    with sr.Microphone() as source:
        print("Listening for name...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            name = recognizer.recognize_google(audio)
            print("Recognized name:", name)
            return name
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print("Error retrieving recognition results; {0}".format(e))
            return None

def save_name(name):
    with open("names.txt", "a") as file:
        file.write(name + "\n")

def load_names():
    names = []
    try:
        with open("names.txt", "r") as file:
            for line in file:
                names.append(line.strip())
    except FileNotFoundError:
        print("Names file not found. Creating new file.")
        open("names.txt", "a").close()
    return names

def speak(text):
    engine.say(text)
    engine.runAndWait()

def capture_images_and_save(name, video_capture):
    print(f"Please look at the camera for capturing images of {name}")
    for i in range(5):
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        image_path = os.path.join("known_faces", f"{name}_{i}.jpg")
        cv2.imwrite(image_path, frame)

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    names = load_names()
    for name in names:
        image_paths = [f for f in os.listdir("known_faces") if f.startswith(name)]
        for image_path in image_paths:
            image = face_recognition.load_image_file(os.path.join("known_faces", image_path))
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]  
                known_face_encodings.append(encoding)
                known_face_names.append(name)
            else:
                print(f"No face found in {image_path}")
    return known_face_encodings, known_face_names

def open_camera():
    video_capture = cv2.VideoCapture(1)
    if not video_capture.isOpened():
        print("Error: Unable to open camera.")
        return None
    return video_capture

known_face_encodings, known_face_names = load_known_faces()

greeted_names = {}
video_capture = open_camera()

while video_capture:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display the camera feed
    cv2.imshow('Video', frame)

    spoken_name = recognize_speech()
    if spoken_name:
        name = spoken_name
        if name in known_face_names:
            if name not in greeted_names:
                speak(f"Hello {name}, welcome back!")
                greeted_names[name] = True
        else:
            speak(f"Hello {name}, nice to meet you! I will remember your face for next time.")
            save_name(name)
            capture_images_and_save(name, video_capture)
            known_face_encodings, known_face_names = load_known_faces()

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

video_capture.release()
cv2.destroyAllWindows()
