import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import load_model
from scipy import stats
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextGenerationModel

# 수화 인식 모델 설정
actions = np.array(['Normal', 'Hi', 'Meet', 'Nice', 'Age', 'How', 'Ten', 'Feeling', 'Good', 'Next'])
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
model = load_model('Model2.h5')

# 수화 인식 관련 함수들
def multiple_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #Color convertion 
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connect


def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    # Extract pose landmarks (landmarks 11 to 22)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for i, res in enumerate(results.pose_landmarks.landmark) if 11 <= i + 1 <= 22]).flatten() if results.pose_landmarks else np.zeros(12*4)

    # Extract left hand landmarks
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    # Extract right hand landmarks
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    # Concatenate and return the results
    return np.concatenate([pose, lh, rh])

colors = [(245,117,16), (117,245,16), (16,117,245),(200,103,27),(245,117,16), (117,245,16), (16,117,245),(245,117,16), (117,245,16), (16,117,245)]

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA) 
        
    return output_frame

# 문장 생성 함수
def mk_sentence(temperature: float, project_id: str, location: str, words: list) -> str:
    vertexai.init(project=project_id, location=location)
    parameters = {
        "temperature": temperature,
        "max_output_tokens": 2048,
        "top_p": 1,
        "top_k": 0,
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")
    prompt = "다음의 단어들을 순서대로 조합해서 자연스러운 문장으로 만들어줘. " + " ".join(words)
    response = model.predict(prompt, **parameters)
    return response.text

# 문장 생성에 필요한 설정
project_id = "striped-strata-411107"
location = "asia-northeast3"
temperature = 0.9

# 수화 인식 및 문장 생성 통합 코드
sequence = []
sentence = []
predictions = []
threshold = 0.8

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = multiple_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-35:]
        
        if len(sequence) == 35:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            
            if np.unique(predictions[-15:])[0] == np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        if len(sentence) > 0 and (cv2.waitKey(10) & 0xFF == ord('s')):  # 's'를 누를 때 문장 생성
            generated_sentence = mk_sentence(temperature=temperature, project_id=project_id, location=location, words=sentence)
            print("Generated Sentence: ", generated_sentence)

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
