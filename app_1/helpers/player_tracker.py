"""
Player Tracker Module
=====================
Oyuncu takibi ve jersey doğrulama sistemi.
"""

from collections import defaultdict


class PlayerTracker:
    """
    Oyuncuları takip eden ve jersey numaralarını doğrulayan sınıf.
    
    Bir oyuncuya numara atamadan önce aynı numaranın 
    birden fazla kez tespit edilmesi gerekir.
    
    Kameradan çıkan oyuncular belirli frame sonra unutulur.
    """
    
    def __init__(self, confirmation_count: int = 3, min_confidence: float = 0.7, 
                 timeout_frames: int = 30):
        """
        PlayerTracker'ı başlat.
        
        Args:
            confirmation_count: Onay için gereken tespit sayısı
            min_confidence: Minimum OCR confidence
            timeout_frames: Oyuncu unutulma süresi (frame)
        """
        self.confirmation_count = confirmation_count
        self.min_confidence = min_confidence
        self.timeout_frames = timeout_frames
        
        # Onaylanmış oyuncular: {track_id: jersey_number}
        self.confirmed_players = {}
        
        # Detection history: {track_id: [(number, confidence), ...]}
        self.detection_history = defaultdict(list)
        
        # Son görülme frame'i: {track_id: frame_num}
        self.last_seen = {}
        
        # Stats
        self.stats = {
            'confirmed': 0,
            'ocr_skipped': 0,
            'ocr_performed': 0,
            'forgotten': 0
        }
    
    def update_seen(self, track_id: int, frame_num: int):
        """Oyuncunun görüldüğünü kaydet."""
        self.last_seen[track_id] = frame_num
    
    def cleanup_old_tracks(self, current_frame: int):
        """
        Uzun süredir görülmeyen track_id'leri temizle.
        
        Args:
            current_frame: Şu anki frame numarası
        """
        to_remove = []
        
        for track_id, last_frame in self.last_seen.items():
            if current_frame - last_frame > self.timeout_frames:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            # Onaylı oyuncuları unut
            if track_id in self.confirmed_players:
                del self.confirmed_players[track_id]
            
            # Detection history'yi temizle
            if track_id in self.detection_history:
                del self.detection_history[track_id]
            
            # Last seen'den kaldır
            del self.last_seen[track_id]
            
            self.stats['forgotten'] += 1
    
    def is_confirmed(self, track_id: int) -> bool:
        """Oyuncu onaylı mı kontrol et."""
        return track_id in self.confirmed_players
    
    def get_jersey(self, track_id: int) -> str:
        """Onaylı oyuncunun jersey numarasını döndür."""
        return self.confirmed_players.get(track_id)
    
    def add_detection(self, track_id: int, number: str, confidence: float, 
                      frame_num: int = 0) -> bool:
        """
        Yeni tespit ekle ve onay kontrolü yap.
        
        Args:
            track_id: Oyuncu track ID
            number: Tespit edilen numara
            confidence: OCR confidence
            frame_num: Frame numarası
            
        Returns:
            newly_confirmed: Bu tespit ile onaylandı mı
        """
        # Görüldüğünü kaydet
        self.update_seen(track_id, frame_num)
        
        # Zaten onaylı ise atla
        if self.is_confirmed(track_id):
            return False
        
        # Confidence kontrolü
        if confidence < self.min_confidence:
            return False
        
        # History'ye ekle
        self.detection_history[track_id].append((number, confidence))
        
        # Onay kontrolü
        return self._check_confirmation(track_id)
    
    def _check_confirmation(self, track_id: int) -> bool:
        """
        Onay kontrolü yap - aynı numara N kez tespit edildi mi?
        
        Returns:
            confirmed: Onaylandı mı
        """
        history = self.detection_history[track_id]
        
        if len(history) < self.confirmation_count:
            return False
        
        # Son N tespitten en sık görülen numarayı bul
        recent = history[-self.confirmation_count * 2:]  # Son 2N tespite bak
        
        # Numara sayımı
        number_counts = defaultdict(int)
        number_confidences = defaultdict(list)
        
        for num, conf in recent:
            number_counts[num] += 1
            number_confidences[num].append(conf)
        
        # En çok görülen numara
        if not number_counts:
            return False
            
        best_number = max(number_counts.keys(), key=lambda x: number_counts[x])
        count = number_counts[best_number]
        
        # N kez görüldü mü?
        if count >= self.confirmation_count:
            avg_conf = sum(number_confidences[best_number]) / len(number_confidences[best_number])
            
            # Onayla
            self.confirmed_players[track_id] = best_number
            self.stats['confirmed'] += 1
            
            print(f"🎽 Oyuncu #{track_id} ONAYLANDI: Jersey #{best_number} (avg conf: {avg_conf:.2f})")
            return True
        
        return False
    
    def reset(self):
        """Tüm verileri sıfırla."""
        self.confirmed_players.clear()
        self.detection_history.clear()
        self.last_seen.clear()
        self.stats = {'confirmed': 0, 'ocr_skipped': 0, 'ocr_performed': 0, 'forgotten': 0}
    
    def get_summary(self) -> dict:
        """Özet istatistikler döndür."""
        return {
            'confirmed_players': len(self.confirmed_players),
            'jersey_numbers': list(self.confirmed_players.values()),
            'active_tracks': len(self.last_seen),
            'stats': self.stats.copy()
        }


# Test kodu
if __name__ == "__main__":
    tracker = PlayerTracker(confirmation_count=3, timeout_frames=10)
    
    # Simüle et
    tracker.add_detection(1, "23", 0.95, frame_num=1)
    tracker.add_detection(1, "23", 0.92, frame_num=2)
    tracker.add_detection(1, "23", 0.88, frame_num=3)  # Onaylandı
    
    print(f"Frame 3 - Oyuncu 1 onaylı: {tracker.is_confirmed(1)}")
    
    # 15 frame sonra cleanup
    tracker.cleanup_old_tracks(current_frame=18)
    print(f"Frame 18 - Oyuncu 1 onaylı: {tracker.is_confirmed(1)}")  # Unutuldu
    
    print(f"\nÖzet: {tracker.get_summary()}")

