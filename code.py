import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import random
from streamlit_webrtc import VideoTransformerBase
import os
import sounddevice as sd
import imgaug.augmenters as iaa
from scipy.ndimage import rotate, zoom
from tensorflow.keras.regularizers import l2


# Constants
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
NUM_LANDMARKS = 21 * 3  # 21 landmarks with x, y, z coordinates
NUM_GESTURES = 5  # 0 to 9
SMOOTHING_FRAMES = 5
GEOMETRIC_THRESHOLDS = [0.08, 0.1, 0.12, 0.15]
DATA_DIR = 'collected_data'
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32


# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Gesture labels
GESTURE_LABELS = [
    "Fist", "Open Hand", "Thumbs Up","Friend","Thumbs Down"
]


# Augmentation Functions for Landmark Data
def augment_landmarks(landmarks, num_augmentations=5):
    """Generate augmented versions of landmark data"""
    augmented_data = []
    
    for _ in range(num_augmentations):
        # Copy original landmarks
        aug_landmarks = landmarks.copy()
        
        # Random scaling (simulate different hand sizes)
        scale_factor = np.random.uniform(0.9, 1.1)
        aug_landmarks = aug_landmarks * scale_factor
        
        # Random translation (simulate different hand positions)
        translation_x = np.random.uniform(-0.05, 0.05)
        translation_y = np.random.uniform(-0.05, 0.05)
        
        # Apply translation to x and y coordinates
        for i in range(0, len(aug_landmarks), 3):
            aug_landmarks[i] += translation_x    # x coordinate
            aug_landmarks[i+1] += translation_y  # y coordinate
        
        # Random noise (simulate detection noise)
        noise_factor = np.random.uniform(0, 0.01)
        noise = np.random.normal(0, noise_factor, aug_landmarks.shape)
        aug_landmarks += noise
        
        # Random rotation simulation (for hand orientation)
        rotation_factor = np.random.uniform(-0.2, 0.2)
        for i in range(0, len(aug_landmarks), 3):
            x, y = aug_landmarks[i], aug_landmarks[i+1]
            aug_landmarks[i] = x * np.cos(rotation_factor) - y * np.sin(rotation_factor)
            aug_landmarks[i+1] = x * np.sin(rotation_factor) + y * np.cos(rotation_factor)
        
        # Random finger flexion (simulate slight variations in pose)
        if random.random() > 0.5:
            for finger_base in [1, 5, 9, 13, 17]:  # Base indices for each finger
                if random.random() > 0.7:  # Only modify some fingers
                    flex_factor = np.random.uniform(0.95, 1.05)
                    for j in range(1, 4):  # Modify the 3 joints of the finger
                        idx = (finger_base + j) * 3
                        # Slightly modify the position relative to previous joint
                        if idx < len(aug_landmarks) - 3:
                            aug_landmarks[idx:idx+2] = aug_landmarks[idx:idx+2] * flex_factor
        
        augmented_data.append(aug_landmarks)
    
    return augmented_data


# Optimization Classes with improved hyperparameter ranges
class AELM:
    def __init__(self, mutation_rate=0.3):
        self.mutation_rate = mutation_rate
        self.optimizers = ['adam', 'rmsprop', 'sgd']
        self.current_optimizer = 'adam'
        self.learning_rates = [0.001, 0.0005, 0.0001]
        self.current_lr = 0.001

    def evolve_optimizer(self, current_accuracy):
        if random.random() < self.mutation_rate:
            self.current_optimizer = random.choice(self.optimizers)
            self.current_lr = random.choice(self.learning_rates)
        return self.current_optimizer, self.current_lr


class MRFO:
    def __init__(self):
        self.best_params = {
            'lr': 0.001,
            'dropout': 0.3,
            'geom_threshold': 0.1,
            'l2_reg': 0.001,
            'batch_norm': True
        }
        
    def optimize_hyperparams(self, current_accuracy=0):
        # If performance is good, make smaller adjustments
        if current_accuracy > 0.9:
            lr = random.choice([0.0005, 0.0001, 0.00005])
            dropout = random.choice([0.2, 0.25, 0.3])
            geom_threshold = random.choice([0.08, 0.09, 0.1])
            l2_reg = random.choice([0.0005, 0.001])
            batch_norm = random.choice([True, True, False])  # Favor batch norm
        else:
            # More exploration for lower accuracy
            lr = random.choice([0.001, 0.0005, 0.0001, 0.01])
            dropout = random.choice([0.2, 0.3, 0.4, 0.5])
            geom_threshold = random.choice(GEOMETRIC_THRESHOLDS)
            l2_reg = random.choice([0.001, 0.01, 0.0001])
            batch_norm = random.choice([True, False])
            
        # Update best params if accuracy improves (handled in HybridGestureRecognizer)
        
        return lr, dropout, geom_threshold, l2_reg, batch_norm


class HybridGestureRecognizer:
    def __init__(self):
        self.hands = mp_hands.Hands(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.templates = defaultdict(list)
        self.geometric_threshold = 0.1
        self.model = None
        self.scaler = StandardScaler()
        self.aelm = AELM()
        self.mrfo = MRFO()
        self.best_accuracy = 0.0
        self.best_params = {}
        # Smoothing for predictions
        self.prediction_history = []
        self.history_size = 3

    def init_model(self, l2_reg=0.001, batch_norm=True):
        optimizer_name, lr = self.aelm.evolve_optimizer(self.best_accuracy)

        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=lr)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=lr)
        else:
            optimizer = SGD(learning_rate=lr)

        model = Sequential()
        
        # Input layer
        model.add(Dense(64, activation='relu', input_shape=(NUM_LANDMARKS,), 
                         kernel_regularizer=l2(l2_reg)))
        
        if batch_norm:
            model.add(BatchNormalization())
            
        model.add(Dropout(0.3))
        
        # Hidden layer 1
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(l2_reg)))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Hidden layer 2 - additional layer for more capacity
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(NUM_GESTURES, activation='softmax'))
        
        model.compile(optimizer=optimizer,
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.model = model

    def add_template(self, gesture_id, landmarks):
        self.templates[gesture_id].append(landmarks)

    def geometric_match(self, landmarks):
        best_distance = float('inf')
        best_gesture = None
        
        for gesture_id, templates in self.templates.items():
            for template in templates:
                distance = np.mean(np.abs(landmarks - template))
                if distance < best_distance:
                    best_distance = distance
                    best_gesture = gesture_id
        
        # Only return if we're confident enough
        if best_distance < self.geometric_threshold:
            return best_gesture
        return None

    def train(self, X, y, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE):
        # Augment training data
        augmented_X = []
        augmented_y = []
        
        for i, landmarks in enumerate(X):
            # Keep original data
            augmented_X.append(landmarks)
            augmented_y.append(y[i])
            
            # Add augmented versions
            aug_data = augment_landmarks(landmarks, num_augmentations=5)
            augmented_X.extend(aug_data)
            augmented_y.extend([y[i]] * len(aug_data))
        
        X = np.array(augmented_X)
        y = np.array(augmented_y)
        
        # Split with stratification to ensure balanced classes
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Normalize data
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)

        # Setup callbacks for training
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]

        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        training_placeholder = st.empty()
        
        # Iterative training with optimization
        for epoch in range(epochs):
            # Get hyperparameters for this iteration
            lr, dropout, self.geometric_threshold, l2_reg, batch_norm = self.mrfo.optimize_hyperparams(self.best_accuracy)
            optimizer_name, _ = self.aelm.evolve_optimizer(self.best_accuracy)
            
            # Initialize model with new hyperparameters
            self.init_model(l2_reg, batch_norm)
            
            # Train for one epoch
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=1,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate and save if improved
            val_accuracy = history.history['val_accuracy'][0]
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.model.save('best_model.h5')
                np.save('scaler_mean.npy', self.scaler.mean_)
                np.save('scaler_scale.npy', self.scaler.scale_)
                # Save best hyperparameters
                self.best_params = {
                    'lr': lr,
                    'dropout': dropout,
                    'geometric_threshold': self.geometric_threshold,
                    'l2_reg': l2_reg,
                    'batch_norm': batch_norm,
                    'optimizer': optimizer_name
                }
            
            # Update progress
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch {epoch+1}/{epochs}, Accuracy: {val_accuracy:.4f}, Best: {self.best_accuracy:.4f}")
            
            # Only show final training status at the end
            if epoch == epochs - 1:
                training_placeholder.success(
                    f"Training complete!\n"
                    f"Best validation accuracy: {self.best_accuracy:.4f}\n"
                    f"Final optimizer: {self.aelm.current_optimizer}\n"
                    f"Best hyperparameters: {self.best_params}"
                )
        
        # Final model with best parameters
        self.init_model(self.best_params.get('l2_reg', 0.001), 
                       self.best_params.get('batch_norm', True))
        self.model = load_model('best_model.h5')
        
        return self.best_accuracy

    def smooth_prediction(self, prediction):
        # Add to history
        self.prediction_history.append(prediction)
        
        # Keep history at fixed size
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Return most common prediction
        if self.prediction_history:
            counts = np.bincount(self.prediction_history)
            return np.argmax(counts)
        else:
            return prediction

    def predict(self, landmarks):
        if landmarks is None:
            return None
            
        gesture = self.geometric_match(landmarks)
        
        if gesture is None and self.model:
            scaled = self.scaler.transform([landmarks])
            pred_probs = self.model.predict(scaled, verbose=0)
            confidence = np.max(pred_probs)
            
            # Only accept predictions with sufficient confidence
            if confidence > 0.7:
                pred_class = np.argmax(pred_probs)
                # Apply temporal smoothing
                return self.smooth_prediction(pred_class)
            else:
                # Add None to history for low confidence predictions
                self.prediction_history.append(None)
                if len(self.prediction_history) > self.history_size:
                    self.prediction_history.pop(0)
                return None
        return gesture

    def get_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Extract x, y, z of 21 landmarks for the first detected hand
            landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark
            ]).flatten()
            return landmarks  # shape: (63,)
        
        return None


class GestureTransformer(VideoTransformerBase):
    def __init__(self, recognizer, stop_streaming, mute_audio=False):
        self.recognizer = recognizer
        self.current_gesture = "None"
        self.stop_streaming = stop_streaming
        self.mute_audio = mute_audio
        self.confidence = 0.0
        if mute_audio:
            sd.stop()  # Stop any audio playback

    def transform(self, frame):
        if self.stop_streaming():
            return frame.to_ndarray(format="bgr24")

        img = frame.to_ndarray(format="bgr24")
        landmarks = self.recognizer.get_landmarks(img)

        if landmarks is not None:
            results = self.recognizer.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                prediction = self.recognizer.predict(landmarks)
                if prediction is not None:
                    self.current_gesture = GESTURE_LABELS[prediction]
                    # Get confidence from model
                    scaled = self.recognizer.scaler.transform([landmarks])
                    pred_probs = self.recognizer.model.predict(scaled, verbose=0)
                    self.confidence = np.max(pred_probs)

        # Display with confidence
        color = (0, 255, 0) if self.confidence > 0.8 else (0, 165, 255)
        cv2.putText(img, f"Gesture: {self.current_gesture} ({self.confidence:.2f})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return img


def main():
    st.title("GESTIFY")
    st.markdown("""
    Advanced Hybrid Gesture Recognition System combining:
    - Geometric template matching (fast)
    - Neural Network with optimization (accurate)
    - Data augmentation for improved robustness
    """)

    recognizer = HybridGestureRecognizer()

    # App Modes
    mode = st.sidebar.selectbox("Select Mode",
                                ["Real-time Recognition", "Data Collection", "Model Training"])

    if mode == "Data Collection":
        st.header("Data Collection")
        st.info("Perform this in a well-lit environment with clear hand gestures.")
        st.info("Press 'c' in the OpenCV window to capture a sample, 'q' to stop collecting for the current gesture.")

        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        gesture_id = st.selectbox("Select Gesture",
                               list(range(len(GESTURE_LABELS))),
                               format_func=lambda x: GESTURE_LABELS[x])

        # Add options for automatic data collection
        auto_collect = st.checkbox("Enable Automatic Collection")
        collection_interval = st.slider("Collection Interval (seconds)", 0.5, 5.0, 1.5) if auto_collect else 0
        max_samples = st.slider("Maximum Samples to Collect", 50, 500, 100) if auto_collect else 0

        if st.button("Start Collection"):
            cap = cv2.VideoCapture(0)
            collected_landmarks = []
            collection_info = st.empty()
            last_collection_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = recognizer.hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    # Draw landmarks for visualization
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    # Get the first hand's landmarks as numpy array
                    landmarks = np.array([[lm.x, lm.y, lm.z]
                                      for lm in results.multi_hand_landmarks[0].landmark]).flatten()
                    
                    # Auto collection logic
                    current_time = time.time()
                    if auto_collect and current_time - last_collection_time >= collection_interval:
                        if len(collected_landmarks) < max_samples:
                            collected_landmarks.append(landmarks)
                            collection_info.info(f"Auto-collecting: {len(collected_landmarks)}/{max_samples} samples.")
                            last_collection_time = current_time
                        else:
                            auto_collect = False
                            collection_info.success(f"Auto-collection complete: {len(collected_landmarks)} samples")

                cv2.putText(frame, f"Collecting: {GESTURE_LABELS[gesture_id]} ({len(collected_landmarks)} samples)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Data Collection - Press 'c' to capture, 'q' to stop", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('c') and results.multi_hand_landmarks:
                    collected_landmarks.append(landmarks)  # Save the numpy array
                    collection_info.info(f"Collected {len(collected_landmarks)} samples.")
                    time.sleep(0.2)  # Small delay between captures

                elif key == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

            if collected_landmarks:
                save_path = os.path.join(DATA_DIR, f"{GESTURE_LABELS[gesture_id].replace(' ', '_')}.npy")
                np.save(save_path, np.array(collected_landmarks))
                st.success(f"Saved {len(collected_landmarks)} samples for {GESTURE_LABELS[gesture_id]}")
                
                # Option to generate augmented samples
                if st.button("Generate Additional Augmented Samples"):
                    all_landmarks = np.array(collected_landmarks)
                    augmented_data = []
                    
                    for landmark in all_landmarks:
                        augmented_data.extend(augment_landmarks(landmark, num_augmentations=10))
                    
                    augmented_save_path = os.path.join(DATA_DIR, f"{GESTURE_LABELS[gesture_id].replace(' ', '_')}_augmented.npy")
                    np.save(augmented_save_path, np.array(augmented_data))
                    st.success(f"Generated and saved {len(augmented_data)} augmented samples")

    elif mode == "Model Training":
        st.header("Model Training")

        collected_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]
        if not collected_files:
            st.warning("No collected data found. Please go to 'Data Collection' mode first.")
            return

        st.info(f"Found collected data for: {[f.replace('.npy', '').replace('_', ' ') for f in collected_files]}")

        # Advanced training options
        st.subheader("Training Configuration")
        use_augmentation = st.checkbox("Use Data Augmentation", value=True)
        augmentation_factor = st.slider("Augmentation Samples per Original", 1, 10, 5) if use_augmentation else 0
        
        advanced_options = st.expander("Advanced Training Options")
        with advanced_options:
            epochs = st.slider("Training Epochs", 10, 200, 50)
            batch_size = st.slider("Batch Size", 16, 128, 32)
            patience = st.slider("Early Stopping Patience", 5, 20, 10)
            val_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
            
            use_early_stopping = st.checkbox("Use Early Stopping", value=True)
            use_lr_scheduler = st.checkbox("Use Learning Rate Scheduler", value=True)
            use_batch_norm = st.checkbox("Use Batch Normalization", value=True)
            l2_strength = st.select_slider("L2 Regularization Strength", 
                                           options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
        if st.button("Load and Prepare Data"):
            all_landmarks = []
            all_labels = []
            
            # Use GESTURE_LABELS for a fixed mapping
            label_map = {label: i for i, label in enumerate(GESTURE_LABELS)}  # Fixed mapping!
            
            for file in collected_files:
                file_path = os.path.join(DATA_DIR, file)
                landmarks = np.load(file_path)
                
                # Extract base label (remove "augmented" suffix if present)
                base_label = file.replace('.npy', '').replace('_', ' ').split(' augmented')[0]
                
                if base_label not in label_map:
                    st.error(f"Error: Gesture '{base_label}' not found in GESTURE_LABELS. Please check your data or GESTURE_LABELS definition.")
                    return
                
                label_id = label_map[base_label]
                
                labels = [label_id] * len(landmarks)
                all_landmarks.extend(landmarks)
                all_labels.extend(labels)

            if all_landmarks:
                st.session_state.train_X = np.array(all_landmarks)
                st.session_state.train_y = np.array(all_labels)
                st.success(f"Loaded {len(all_landmarks)} samples for training.")
                
                # Show class distribution
                unique_labels, counts = np.unique(st.session_state.train_y, return_counts=True)
                distribution = {}
                for label, count in zip(unique_labels, counts):
                    if 0 <= label < len(GESTURE_LABELS):
                        distribution[GESTURE_LABELS[label]] = count
                    else:
                        st.error(f"Label {label} is out of the expected range (0-{len(GESTURE_LABELS) - 1}). Please check your data.")
                        return  # Stop further processing if there's an invalid label
                st.write("Class Distribution:", distribution)
            else:
                st.warning("No landmark data loaded.")

        if 'train_X' in st.session_state and 'train_y' in st.session_state:
            X = st.session_state.train_X
            y = st.session_state.train_y
            
            if st.button("Start Training"):
                # Initialize recognizer with selected options
                recognizer.mrfo.best_params['batch_norm'] = use_batch_norm
                recognizer.mrfo.best_params['l2_reg'] = l2_strength
                
                # Setup callbacks based on user selections
                callbacks = []
                if use_early_stopping:
                    callbacks.append(EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True))
                if use_lr_scheduler:
                    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5))
                
                with st.spinner("Training in progress..."):
                    accuracy = recognizer.train(X, y, epochs, batch_size)

                st.success(f"Training complete! Best validation accuracy: {accuracy:.2%}")

                # Basic training evaluation
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                recognizer.model = load_model('best_model.h5')
                recognizer.scaler.mean_ = np.load('scaler_mean.npy')
                recognizer.scaler.scale_ = np.load('scaler_scale.npy')
                X_test_scaled = recognizer.scaler.transform(X_test)
                _, accuracy = recognizer.model.evaluate(X_test_scaled, y_test, verbose=0)
                st.subheader("Test Evaluation")
                st.info(f"Test Accuracy: {accuracy:.2%}")
                
                # Show best hyperparameters
                st.subheader("Best Hyperparameters")
                st.json(recognizer.best_params)

        else:
            st.info("Load and prepare data first.")

    else:  # Real-time Recognition
        st.header("Real-time Gesture Recognition")
        
        # Check if model exists
        if not os.path.exists('best_model.h5') or not os.path.exists('scaler_mean.npy') or not os.path.exists('scaler_scale.npy'):
            st.warning("Please train a model first in the 'Model Training' mode.")
            return
            
        trained_gestures = []
        if os.path.exists(DATA_DIR):
            trained_gestures = [f.replace('.npy', '').replace('_', ' ').split(' augmented')[0]
                           for f in os.listdir(DATA_DIR)
                           if f.endswith('.npy')]
            # Remove duplicates
            trained_gestures = list(set(trained_gestures))
        
        if not trained_gestures:
            st.warning("No gesture data found. Please collect data first.")
            return
            
        # Load trained model
        try:
            recognizer.model = load_model('best_model.h5')
            recognizer.scaler.mean_ = np.load('scaler_mean.npy')
            recognizer.scaler.scale_ = np.load('scaler_scale.npy')
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        # Recognition options
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7)
        use_smoothing = st.checkbox("Use Temporal Smoothing", value=True)
        smoothing_frames = st.slider("Smoothing Window Size", 1, 10, 3) if use_smoothing else 1
        recognizer.history_size = smoothing_frames
        
        # Start/Stop button
        if st.button("Start Recognition"):
            cap = cv2.VideoCapture(0)
            stop_recognition = False
            
            # Create a placeholder for status messages
            status_placeholder = st.empty()
            confidence_placeholder = st.empty()
            status_placeholder.info(f"Detecting gestures: {', '.join(trained_gestures)}. Press 'q' to stop.")

            while cap.isOpened() and not stop_recognition:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Mirror the frame
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process hand landmarks
                results = recognizer.hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    # Draw landmarks
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    # Extract landmarks and predict gesture
                    landmarks = np.array([[lm.x, lm.y, lm.z]
                                       for lm in results.multi_hand_landmarks[0].landmark]).flatten()
                    
                    # Scale and predict
                    scaled_landmarks = recognizer.scaler.transform([landmarks])
                    prediction = recognizer.model.predict(scaled_landmarks, verbose=0)
                    gesture_id = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    confidence_placeholder.info(f"Confidence: {confidence:.2f}")
                    
                    # Apply smoothing and confidence thresholding
                    # Apply smoothing and confidence thresholding
                    if confidence >= confidence_threshold:
                        if use_smoothing:
                            # Add to prediction history for temporal smoothing
                            recognizer.prediction_history.append(gesture_id)
                            if len(recognizer.prediction_history) > smoothing_frames:
                                recognizer.prediction_history.pop(0)
                            
                            # Get most common prediction in the window
                            counts = np.bincount(recognizer.prediction_history)
                            smoothed_gesture_id = np.argmax(counts)
                            predicted_gesture = GESTURE_LABELS[smoothed_gesture_id]
                        else:
                            predicted_gesture = GESTURE_LABELS[gesture_id]
                        
                        # Display with color coded confidence
                        if confidence > 0.9:
                            color = (0, 255, 0)  # Green for high confidence
                        elif confidence > 0.7:
                            color = (0, 165, 255)  # Orange for medium confidence
                        else:
                            color = (0, 0, 255)  # Red for lower confidence
                            
                        cv2.putText(frame, f"{predicted_gesture} ({confidence:.2f})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    else:
                        cv2.putText(frame, "Uncertain Gesture",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                # Show the frame
                cv2.imshow("Real-time Gesture Recognition - Press 'q' to quit", frame)
                
                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stop_recognition = True
                    break

            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            status_placeholder.success("Gesture recognition stopped.")


if __name__ == "__main__":
    main()
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
