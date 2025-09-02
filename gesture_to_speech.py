import cv2
import mediapipe as mp
import pyttsx3
import time
import numpy as np

class GestureToSpeech:
    def __init__(self):
        # Inisialisasi MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Inisialisasi Text-to-Speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Kecepatan bicara
        self.tts_engine.setProperty('volume', 0.8)  # Volume suara
        
        # Mapping gesture ke teks
        self.gesture_mapping = {
            'thumbs_up': "Halo",
            'peace': "Nama saya",
            'wave': "Akbar Rayyan Al Ghifari"
        }
        
        # Variabel untuk optimasi
        self.last_gesture_time = 0
        self.cooldown_duration = 2.0  # 2 detik cooldown
        self.last_detected_gesture = None
        
        # Variabel untuk tracking
        self.gesture_confidence_threshold = 0.8
        self.gesture_frames_needed = 5  # Frame yang dibutuhkan untuk konfirmasi
        self.gesture_frame_count = 0
        self.current_gesture = None
        
    def is_finger_extended(self, landmarks, finger_tips, finger_pips):
        """Mengecek apakah jari dalam keadaan terbuka"""
        extended = []
        
        # Thumb (jempol) - perhitungan khusus
        if landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x:
            extended.append(True)
        else:
            extended.append(False)
            
        # Jari lainnya
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_pips[i]].y:
                extended.append(True)
            else:
                extended.append(False)
                
        return extended
    
    def detect_gesture(self, landmarks):
        """Mendeteksi gesture berdasarkan landmarks tangan"""
        # Titik-titik ujung jari dan sendi
        finger_tips = [4, 8, 12, 16, 20]  # Ujung jari
        finger_pips = [3, 6, 10, 14, 18]  # Sendi jari
        
        # Cek apakah jari terbuka
        extended_fingers = self.is_finger_extended(landmarks, finger_tips, finger_pips)
        
        # Deteksi gesture berdasarkan pola jari
        if extended_fingers == [True, False, False, False, False]:
            return 'thumbs_up'  # 👍
        elif extended_fingers == [False, True, True, False, False]:
            return 'peace'  # ✌
        elif extended_fingers == [False, True, True, True, True]:
            return 'wave'  # 👋 (semua jari kecuali jempol)
        else:
            return 'unknown'
    
    def process_gesture(self, gesture):
        """Memproses gesture yang terdeteksi dengan optimasi"""
        current_time = time.time()
        
        # Reset counter jika gesture berubah
        if gesture != self.current_gesture:
            self.current_gesture = gesture
            self.gesture_frame_count = 0
        
        # Increment counter untuk gesture yang sama
        if gesture != 'unknown':
            self.gesture_frame_count += 1
            
            # Cek apakah sudah cukup frame untuk konfirmasi
            if (self.gesture_frame_count >= self.gesture_frames_needed and 
                gesture != self.last_detected_gesture and
                current_time - self.last_gesture_time > self.cooldown_duration):
                
                # Konfirmasi gesture dan keluarkan suara
                if gesture in self.gesture_mapping:
                    text = self.gesture_mapping[gesture]
                    print(f"Gesture terdeteksi: {gesture} -> '{text}'")
                    
                    # Text-to-Speech
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    
                    # Update tracking variables
                    self.last_detected_gesture = gesture
                    self.last_gesture_time = current_time
        else:
            # Reset jika tidak ada gesture yang valid
            self.gesture_frame_count = 0
    
    def run(self):
        """Fungsi utama untuk menjalankan sistem"""
        # Inisialisasi webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Tidak dapat membuka webcam")
            return
        
        print("Sistem Gesture to Speech dimulai!")
        print("Gesture yang didukung:")
        print("👍 (Thumbs Up) -> 'Halo'")
        print("✌ (Peace) -> 'Nama saya'")
        print("👋 (Wave) -> 'Akbar Rayyan Al Ghifari'")
        print("Tekan 'q' untuk keluar")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame dari webcam")
                break
            
            # Flip frame horizontal untuk mirror effect
            frame = cv2.flip(frame, 1)
            
            # Konversi BGR ke RGB untuk MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Proses deteksi tangan
            results = self.hands.process(rgb_frame)
            
            # Gambar landmarks jika tangan terdeteksi
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Gambar landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Deteksi gesture
                    gesture = self.detect_gesture(hand_landmarks.landmark)
                    
                    # Proses gesture
                    self.process_gesture(gesture)
                    
                    # Tampilkan gesture di frame
                    cv2.putText(frame, f"Gesture: {gesture}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Tampilkan frame
            cv2.imshow('Gesture to Speech', frame)
            
            # Keluar jika tombol 'q' ditekan
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Sistem dihentikan.")

def main():
    """Fungsi main untuk menjalankan aplikasi"""
    try:
        gesture_system = GestureToSpeech()
        gesture_system.run()
    except KeyboardInterrupt:
        print("\nSistem dihentikan oleh user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
