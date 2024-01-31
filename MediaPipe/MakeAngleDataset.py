import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils #Draw utilities 
no_sequences = 10
sequence_length = 25
start_folder =0
# actions = np.array(['Hi', 'Meet', 'Break Up','Nice','Normal','Crying','Me','You'])    # 가변
actions = np.array(['Hi', 'Meet', 'Break Up','Nice','Normal','Me','You'])    # 가변


DATA_PATH = os.path.join(os.getcwd(),'MP_Data7') 
# # 필요하다면..
# os.makedirs(os.path.join(DATA_PATH))

# 각 Action의 Info를 저장할 폴더가 없다면
for action in actions: 
    # os.makedirs(os.path.join(DATA_PATH,action))
    
    for sequence in range(0,no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


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

def multiple_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #Color convertion 
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def extract_keypoints(results):
    # Extract pose landmarks (landmarks 11 to 22)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for i, res in enumerate(results.pose_landmarks.landmark) if 11 <= i + 1 <= 22]).flatten() if results.pose_landmarks else np.zeros(12*4)
    # Extract left hand landmarks
    lhn = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    # Extract right hand landmarks
    rhn = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    if results.left_hand_landmarks:
        joint1 = np.zeros((21, 4))
        for j, lm in enumerate(results.left_hand_landmarks.landmark):
            joint1[j] = [lm.x, lm.y, lm.z, lm.visibility]

        # Compute angles between joints
        v1 = joint1[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
        v2 = joint1[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
        v = v2 - v1 # [20, 3]
        # Normalize v
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

        angle = np.degrees(angle) # Convert radian to degree

        lh = np.array(angle, dtype=np.float32)
    else:
        lh = np.zeros(15)

    if results.right_hand_landmarks:
        joint2 = np.zeros((21, 4))
        for j, lm in enumerate(results.right_hand_landmarks.landmark):
            joint2[j] = [lm.x, lm.y, lm.z, lm.visibility]

        # Compute angles between joints
        v12 = joint2[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
        v22 = joint2[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
        v2 = v22 - v12 # [20, 3]
        # Normalize v
        v2 = v2 / np.linalg.norm(v2, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle2 = np.arccos(np.einsum('nt,nt->n',
            v2[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v2[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

        angle2 = np.degrees(angle2) # Convert radian to degree

        rh = np.array(angle2, dtype=np.float32)
    else:
        rh = np.zeros(15)

    print("pose: ",pose.shape," lh: ",lh.shape," rh: ",rh.shape)
    # Concatenate and return the results
    return np.concatenate([pose, lh,lhn, rh,rhn])

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connect

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for action in actions:
        for sequence in range(start_folder, start_folder+no_sequences):
            for frame_num in range(sequence_length):

                ret, frame = cap.read()
                image, results = multiple_detection(frame, holistic)

                draw_styled_landmarks(image, results)
                
                if frame_num == 0: 
                    cv2.putText(image,action + "  " + "Collecting", (150,300), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (100,100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()