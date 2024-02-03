import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from scipy import stats

log_dir = os.path.join(os.getcwd(),'Logs')
actions = np.array(['Normal', 'Hi', 'Meet','Nice','Age','How','Ten','Feeling','Good','Next'])    # 가변

# 영어 단어를 한글 단어로 매핑하는 사전
word_map = {
    'Hi': '안녕',
    'Meet': '만나다',
    'Nice': '좋다',
    'Age': '나이',
    'How': '어떻게',
    'Ten': '10',
    'Feeling': '느낌',
    'Good': '좋다',
    'Next': '다음'
}

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def multiple_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results
    
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for i, res in enumerate(results.pose_landmarks.landmark) if 11 <= i + 1 <= 22]).flatten() if results.pose_landmarks else np.zeros(12*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

model = load_model('./Model2.h5')

colors = [(245,117,16), (117,245,16), (16,117,245),(200,103,27),(245,117,16), (117,245,16), (16,117,245),(245,117,16), (117,245,16), (16,117,245)]

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA) 
    return output_frame

# New code to load an mp4 file
video_path = '/home/autonav/MediaTest/Hi.mp4'
cap = cv2.VideoCapture(video_path)

sequence = []
sentence = []
predictions = []
Output = []
threshold = 0.8

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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
                sentence = sentence[-1:]

            image = prob_viz(res, actions, image, colors)

        if Output == [] and sentence != "normal":
            Output.append(sentence)
        elif Output != [] and sentence != "normal":
            if Output[-1] != sentence:
                Output.append(sentence)

        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
Output = [item for sublist in Output for item in sublist]

# Output 리스트의 단어들을 한글 단어로 변환
Output_korean = [word_map[word] if word in word_map else word for word in Output]

from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextGenerationModel

project_id = "sibal-413208"
location = "asia-northeast3"
temperature = 0.9

def mk_sentence(
    temperature: float,
    project_id: str,
    location: str,
    words: list  # Initialize the words list
) -> str:
    """Ideation example with a Large Language Model"""

    vertexai.init(project=project_id, location=location)
    # TODO developer - override these parameters as needed:
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 1024,  # Token limit determines the maximum amount of text output.
        "top_p": 1,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 0,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }
    # 평탄화 과정 - 중첩 리스트를 단일 리스트로 변환
    # flat_list = [item for sublist in words for item in sublist] if words and isinstance(words[0], list) else words

    model = TextGenerationModel.from_pretrained("text-bison@001")
    # Combine the fixed prompt with the words list
    prompt = "다음의 단어들을 순서대로 이용해서해서 자연스러운 대화형 문장 하나로 만들어줘. 대화형이라고 해서 상대의 대답까지 만들어달라는건 아니야. 예를 들어 주어진 단어가(나 오늘 금요일 기분 좋다)라면 (나는 오늘이 금요일이라 기분이 좋아)라고 자연스럽게 만들어줘. 이때 마지막이 물음표면 의문문으로 만들어줘. " + ",".join(words)
    response = model.predict(
        prompt,
        **parameters,
    )
    print(f"Response from Model: {response.text}")

    return response.text

# 변환된 한글 단어 리스트를 mk_sentence 함수에 넣어 실행
if __name__ == "__main__":
    mk_sentence(temperature=temperature, project_id=project_id, location=location, words=Output_korean)