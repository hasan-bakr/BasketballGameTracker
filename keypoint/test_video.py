from ultralytics import YOLO
import cv2
import numpy as np

# Model yolu
model = YOLO(r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\models\keypoints\best.pt")

# Video yolu
video_path = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\input\court.mp4"
cap = cv2.VideoCapture(video_path)

# Alanlar ve keypoint indexleri
areas = {
    "three_point": [2,3,4,5],
    "hoop_back_line": [0,1,2,5,6,7]
}

# Renkler
area_colors = {
    "three_point": (0, 0, 255),       # kırmızı
    "hoop_back_line": (0, 255, 0)     # yeşil
}

def remove_nearby_keypoints(kps, threshold=15):
    """ Çok yakın keypointleri sil """
    if len(kps) == 0:
        return kps
    keep = []
    for kp in kps:
        if all(np.linalg.norm(kp - np.array(kp2)) >= threshold for kp2 in keep):
            keep.append(kp)
    return np.array(keep)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.95)

    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            kps_xy = result.keypoints.xy.cpu().numpy().reshape(-1, 2)
            kps_conf = result.keypoints.conf.cpu().numpy().reshape(-1)

            # confidence filtresi
            kps_xy = kps_xy[kps_conf > 0.3]

            # her alan için segmentasyon çiz
            for area_name, indices in areas.items():
                selected_kps = np.array([kps_xy[i] for i in indices if i < len(kps_xy)])
                # çok yakın keypointleri temizle
                selected_kps = remove_nearby_keypoints(selected_kps, threshold=15)

                if len(selected_kps) >= 2:
                    # polygon çizimi
                    pts = selected_kps.reshape((-1, 1, 2)).astype(np.int32)
                    isClosed = True if area_name != "three_point" else False
                    cv2.polylines(frame, [pts], isClosed, area_colors[area_name], 2)

                    # keypointleri çiz ve index göster
                    for idx, kp in zip(indices, selected_kps):
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(frame, (x, y), 5, area_colors[area_name], -1)
                        cv2.putText(frame, str(idx), (x+3, y-3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, area_colors[area_name], 1)

    cv2.imshow("Segmented Keypoints", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
