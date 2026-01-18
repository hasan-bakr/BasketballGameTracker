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
    """
    
    def __init__(self, confirmation_count: int = 3, min_confidence: float = 0.7):
        """
        PlayerTracker'ı başlat.
        
        Args:
            confirmation_count: Onay için gereken tespit sayısı
            min_confidence: Minimum OCR confidence
        """
        self.confirmation_count = confirmation_count
        self.min_confidence = min_confidence
        
        # Onaylanmış oyuncular: {track_id: jersey_number}
        self.confirmed_players = {}
        
        # Detection history: {track_id: [(number, confidence), ...]}
        self.detection_history = defaultdict(list)
        
        # Stats
        self.stats = {
            'confirmed': 0,
            'ocr_skipped': 0,
            'ocr_performed': 0
        }
    
    def is_confirmed(self, track_id: int) -> bool:
        """Oyuncu onaylı mı kontrol et."""
        return track_id in self.confirmed_players
    
    def get_jersey(self, track_id: int) -> str:
        """Onaylı oyuncunun jersey numarasını döndür."""
        return self.confirmed_players.get(track_id)
    
    def add_detection(self, track_id: int, number: str, confidence: float) -> bool:
        """
        Yeni tespit ekle ve onay kontrolü yap.
        
        Args:
            track_id: Oyuncu track ID
            number: Tespit edilen numara
            confidence: OCR confidence
            
        Returns:
            newly_confirmed: Bu tespit ile onaylandı mı
        """
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
        self.stats = {'confirmed': 0, 'ocr_skipped': 0, 'ocr_performed': 0}
    
    def get_summary(self) -> dict:
        """Özet istatistikler döndür."""
        return {
            'confirmed_players': len(self.confirmed_players),
            'jersey_numbers': list(self.confirmed_players.values()),
            'stats': self.stats.copy()
        }


# Test kodu
if __name__ == "__main__":
    tracker = PlayerTracker(confirmation_count=3)
    
    # Simüle et
    tracker.add_detection(1, "23", 0.95)
    tracker.add_detection(1, "23", 0.92)
    tracker.add_detection(1, "23", 0.88)  # 3. tespit -> onaylanmalı
    
    tracker.add_detection(2, "10", 0.90)
    tracker.add_detection(2, "10", 0.85)
    # Oyuncu 2 henüz onaylı değil
    
    print(f"\nOyuncu 1 onaylı: {tracker.is_confirmed(1)} -> #{tracker.get_jersey(1)}")
    print(f"Oyuncu 2 onaylı: {tracker.is_confirmed(2)}")
    print(f"\nÖzet: {tracker.get_summary()}")
