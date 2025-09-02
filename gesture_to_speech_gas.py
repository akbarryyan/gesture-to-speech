import cv2
import mediapipe as mp
from gtts import gTTS
import pygame
import time
import numpy as np
import threading
from io import BytesIO

class GestureToSpeechGas:
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
        
        # Inisialisasi Text-to-Speech dengan gTTS
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.is_speaking = False
        except Exception as e:
            print(f"Error inisialisasi pygame mixer: {e}")
            self.is_speaking = False
        
        # Mapping gesture ke teks
        self.gesture_mapping = {
            'thumbs_up': "ok gas",
            'one_finger': "ok gas nomer dua",
            'wave': "torang gas"
        }
        
        # Variabel untuk optimasi
        self.last_gesture_time = 0
        self.cooldown_duration = 1.0  # 1 detik cooldown
        
        # Variabel untuk tracking
        self.gesture_confidence_threshold = 0.8
        self.gesture_frames_needed = 3  # Frame yang dibutuhkan untuk konfirmasi
        self.gesture_frame_count = 0
        self.current_gesture = None
        
    def _speak_thread(self, text):
        """Thread untuk menjalankan TTS"""
        try:
            print(f"Generating audio untuk: '{text}'")
            
            # Generate audio dengan gTTS langsung ke memory
            tts = gTTS(text=text, lang='id', slow=False)
            
            # Simpan audio ke BytesIO (memory)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)  # Reset pointer ke awal
            
            print("Audio generated in memory")
            
            # Putar audio langsung dari memory
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()
            
            # Tunggu sampai audio selesai
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            print("Audio playback completed")
            
        except Exception as e:
            print(f"Error dalam TTS thread: {e}")
        finally:
            self.is_speaking = False
    
    def speak_text(self, text):
        """Menggunakan gTTS untuk text-to-speech"""
        if self.is_speaking:
            return  # Skip jika sedang berbicara
            
        self.is_speaking = True
        
        # Jalankan TTS di thread terpisah
        tts_thread = threading.Thread(target=self._speak_thread, args=(text,))
        tts_thread.daemon = True
        tts_thread.start()
        
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
        elif extended_fingers == [False, True, False, False, False]:
            return 'one_finger'  # 👆 (telunjuk saja)
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
            
            # Cek apakah sudah cukup frame untuk konfirmasi dan cooldown sudah selesai
            if (self.gesture_frame_count >= self.gesture_frames_needed and 
                current_time - self.last_gesture_time > self.cooldown_duration):
                
                # Konfirmasi gesture dan keluarkan suara
                if gesture in self.gesture_mapping:
                    text = self.gesture_mapping[gesture]
                    print(f"Gesture terdeteksi: {gesture} -> '{text}'")
                    
                    # Text-to-Speech dengan gTTS
                    self.speak_text(text)
                    
                    # Update tracking variables
                    self.last_gesture_time = current_time
                    # Reset counter dan current gesture agar bisa diulang
                    self.gesture_frame_count = 0
                    self.current_gesture = None
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
        
        print("Sistem Gesture to Speech GAS dimulai!")
        print("Gesture yang didukung:")
        print("👍 (Thumbs Up) -> 'ok gas'")
        print("👆 (One Finger) -> 'ok gas nomer dua'")
        print("👋 (Wave) -> 'torang gas'")
        print("Tekan 'q' untuk keluar")
        
        # Test pygame mixer
        try:
            pygame.mixer.music.get_busy()
            print("✅ Pygame mixer berfungsi dengan baik")
        except Exception as e:
            print(f"❌ Error pygame mixer: {e}")
            print("Coba install ulang pygame: pip install pygame")
        
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
                    
                    # Tampilkan hasil yang diucapkan saja
                    if gesture in self.gesture_mapping:
                        text = self.gesture_mapping[gesture]
                        cv2.putText(frame, f"'{text}'", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Tampilkan frame
            cv2.imshow('Gesture to Speech GAS', frame)
            
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
        gesture_system = GestureToSpeechGas()
        gesture_system.run()
    except KeyboardInterrupt:
        print("\nSistem dihentikan oleh user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
