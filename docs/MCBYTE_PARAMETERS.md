# MCByte Parametreleri

Bu doküman projede MCByte tracking davranışını etkileyen parametreleri açıklar. Amaç hangi parametrenin neyi değiştirdiğini, hangi yönde oynatmanın ne sonuç doğurabileceğini ve debug loglarında neye bakılması gerektiğini netleştirmektir.

## Kısa Akış

Pipeline önce RF-DETR ile player/referee detection üretir. Sonra bu detectionlar MCByte'a verilir.

MCByte içinde temel sıra şöyledir:

1. Detection skorları `track_thresh` üstü ve altı olarak ayrılır.
2. Yüksek skorlu detectionlar ile mevcut/lost trackler eşleştirilir. Bu `assoc1` adımıdır.
3. İlk adımda eşleşemeyen aktif trackler, düşük skorlu detectionlarla tekrar denenir. Bu `assoc2` adımıdır.
4. Yeni oluşmuş ama henüz kesinleşmemiş trackler kalan detectionlarla denenir. Bu `unconfirmed` adımıdır.
5. Hala kalan detectionlar yeterince güvenliyse yeni track olarak başlatılır.
6. Mask varsa MCByte association cost'unu destekler; bizim wrapper ayrıca duplicate mask ve referee/player conflict temizliği yapar.

## Ana Detection Parametresi

### `--confidence`

Varsayılan: `0.4`

RF-DETR detection thresholdudur. MCByte'a girecek detectionları daha tracker'a ulaşmadan filtreler.

Bu parametre MCByte'ın kendi `track_thresh` değeri değildir. Düşük tutmak MCByte'ın low-score association yapmasına izin verir.

Artırınca:

- Daha az false positive detection gelir.
- Ama occlusion sırasında zayıflayan gerçek oyuncu kutuları kaybolabilir.
- ID kopması artabilir.

Azaltınca:

- Tracker daha fazla kutu görür.
- Low-score association daha iyi çalışabilir.
- False positive ve duplicate track riski artabilir.

Önerilen aralık:

- `0.35 - 0.45` genelde tracking için daha sağlıklı.
- Çok fazla yanlış detection varsa `0.5` denenebilir.

Debug'da bak:

- `[det-debug] raw_players`
- `[det-debug] filtered_players`
- `[mcbyte-assoc:player] input`

## MCByte Core Parametreleri

### `--track-thresh`

Varsayılan: `0.6`

MCByte'ın high-score detection eşiğidir. `score > track_thresh` olan detectionlar ilk association adımına girer.

Artırınca:

- İlk association daha güvenilir detectionlarla yapılır.
- Yanlış match riski azalabilir.
- Ama bazı gerçek oyuncular low-score tarafına düşer veya hiç toparlanamaz.

Azaltınca:

- Daha fazla detection ilk association'a girer.
- ID kopması azalabilir.
- Ama yakın oyuncularda yanlış eşleşme riski artabilir.

Önerilen aralık:

- `0.55 - 0.65`
- Çok kırılganlık varsa `0.55`
- Çok yanlış match varsa `0.65`

Debug'da bak:

- `[mcbyte-assoc:player] high=... low=...`
- High çok düşükse `track_thresh` fazla yüksek olabilir.

### `--new-track-thresh`

Varsayılan: `0.7`

Kalan detectionların yeni track olarak başlatılması için gereken minimum skordur. Upstream MCByte normalde bunu `track_thresh + 0.1` yapıyordu; projede ayrı parametre yaptık.

Artırınca:

- Yeni ID açma zorlaşır.
- Duplicate ID azalabilir.
- Ama gerçekten yeni görünen oyuncu geç başlar veya hiç başlamaz.

Azaltınca:

- Yeni trackler daha kolay açılır.
- Eksik oyuncu daha hızlı görünür.
- Duplicate ve kısa ömürlü ID artabilir.

Önerilen aralık:

- `0.7 - 0.8`
- Çok fazla yeni ID varsa `0.75`
- Oyuncu hiç başlamıyorsa `0.65`

Debug'da bak:

- `[mcbyte-assoc:player] init accepted=... rejected=...`
- Çok `accepted` ve sonra hemen `id-lost` varsa eşik düşük olabilir.
- Çok `rejected` ve oyuncu eksikse eşik yüksek olabilir.

### `--track-buffer`

Varsayılan: `90`

Lost tracklerin kaç frame tutulacağını belirler. Kodda FPS'e göre normalize edilir:

```text
max_time_lost = frame_rate / 30 * track_buffer
```

Artırınca:

- Occlusion sonrası eski ID'ye dönme şansı artar.
- ID kopması azalabilir.
- Ama uzun süre kayıp track yanlış kişiye re-activate olabilir.

Azaltınca:

- Eski trackler daha hızlı silinir.
- Yanlış reactivation azalabilir.
- Ama occlusion sonrası yeni ID açma artabilir.

Önerilen aralık:

- `60 - 120`
- Basketbol gibi hızlı ve kalabalık sahnede `90` iyi başlangıç.

Debug'da bak:

- `[mcbyte-assoc:player] state_before lost=...`
- Çok fazla lost birikiyorsa buffer veya matching fazla gevşek olabilir.

### `--cmc-method`

Varsayılan: `orb`

Camera Motion Compensation metodudur. Kamera hareketini tahmin edip Kalman prediction'ı düzeltmeye çalışır.

Seçenekler:

- `orb`
- `sift`
- `ecc`
- `sparseOptFlow`
- `none`

Artıları/eksileri:

- `orb`: hızlı, default. Spor yayınlarında bazen saha çizgileri/seyirci hareketi yüzünden hatalı transform üretebilir.
- `sift`: daha sağlam olabilir ama daha pahalıdır.
- `sparseOptFlow`: hareketli kamera için denenebilir.
- `ecc`: bazı sahnelerde hassas ama yavaş ve kırılgan olabilir.
- `none`: kamera compensation kapalı. Eğer CMC yanlış yönlendiriyorsa en temiz test budur.

Önerilen test:

```bash
--cmc-method orb
--cmc-method none
```

Debug'da bak:

- `cmc=...` run başında loglanır.
- CMC kaynaklı sorunlar genelde aynı detectionlarla farklı ID davranışı üretir.

## Association Thresholdları

MCByte cost mantığında düşük cost daha iyi eşleşmedir. Bu eşikler maksimum kabul edilebilir cost değerleridir.

Eşik yükselirse matching daha gevşek olur. Eşik düşerse matching daha seçici olur.

### `--assoc1-thresh`

Varsayılan: `0.8`

Birinci association adımının maksimum cost eşiğidir. Bu adım mevcut/lost trackleri high-score detectionlarla eşleştirir.

Artırınca:

- Track kopması azalabilir.
- Ama yakın oyuncular arasında yanlış match riski artabilir.

Azaltınca:

- Yanlış match azalabilir.
- Ama track daha kolay lost olur.
- Fazla düşürülürse sistem kırılganlaşır.

Pratik not:

- `0.7` bazı sahnelerde fazla sıkı olabilir.
- `0.75 - 0.85` daha mantıklı deneme aralığı.

Debug'da bak:

- `[mcbyte-assoc:player] step=1 matches=... unmatched_tracks=...`
- `matched_max` eşik civarına yaklaşıyorsa o frame'de matching sınırda demektir.

### `--assoc2-thresh`

Varsayılan: `0.5`

İkinci association adımının maksimum cost eşiğidir. İlk adımda eşleşemeyen aktif trackler, low-score detectionlarla burada denenir.

Artırınca:

- Düşük skorlu detection ile track kurtarma ihtimali artar.
- Yanlış düşük skorlu kutuyla match riski artar.

Azaltınca:

- Low-score recovery daha seçici olur.
- Yanlış toparlama azalabilir.
- Ama occlusion sonrası track daha kolay lost olur.

Önerilen aralık:

- `0.45 - 0.6`

Debug'da bak:

- `[mcbyte-assoc:player] step=2 pool=... dets=... matches=...`
- `step=2 min=0.8` gibi yüksek cost varsa eşik artırmak mantıklı değildir; detection zaten uzaktadır.

### `--unconfirmed-assoc-thresh`

Varsayılan: `0.7`

Yeni başlatılmış ama henüz güvenilir hale gelmemiş tracklerin kalan detectionlarla eşleşme eşiğidir.

Artırınca:

- Yeni track daha kolay devam eder.
- Yanlış yeni track daha kolay kalıcılaşabilir.

Azaltınca:

- Kısa ömürlü duplicate trackler daha hızlı temizlenir.
- Ama gerçekten yeni track de silinebilir.

Önerilen aralık:

- `0.6 - 0.75`

Debug'da bak:

- `[mcbyte-assoc:player] step=3`
- Yeni ID açılıyor ve hemen kayboluyorsa bu alan önemlidir.

## Mask Parametreleri

Projede player maskleri açık, referee maskleri varsayılan kapalıdır.

```python
use_player_masks = True
use_referee_masks = False
```

Maskler iki yerde kullanılır:

1. MCByte association cost'unu desteklemek için.
2. Bizim wrapper'da duplicate/conflict temizliği için.

### `--mask-duplicate-min-fill`

Varsayılan: `0.45`

Aynı dominant mask'i kullanan birden fazla track varsa duplicate suppression devreye girer. Bu değer, track bbox'ının dominant mask ile ne kadar dolu olması gerektiğini belirler.

Artırınca:

- Duplicate suppression daha az çalışır.
- Gerçek oyuncuyu yanlışlıkla düşürme riski azalır.
- Ama duplicate ID'ler daha fazla kalabilir.

Azaltınca:

- Duplicate ID daha agresif temizlenir.
- Ama Cutie mask iki oyuncuya yayıldığında gerçek track de düşebilir.

Önerilen aralık:

- `0.45 - 0.65`
- Şu an maskler güvenilmezse `0.6` daha güvenli olabilir.

Debug'da bak:

- `[mask-duplicate] frame=... mask=... keep=... drop=... drop_fill=...`
- Çok sık görüyorsan suppression fazla agresif olabilir veya mask drift ediyordur.

### `--ref-player-conflict-iou`

Varsayılan: `0.45`

Referee ile player bbox çakışması bu IoU üstündeyse, ayrıca mask/conf şartları sağlanıyorsa player track düşürülebilir.

Artırınca:

- Player daha zor düşer.
- Yanlış ref/player conflict azalır.
- Referee üstüne binen gerçek player kalabilir.

Azaltınca:

- Referee-player çakışmaları daha agresif temizlenir.
- Yakın geçen oyuncular yanlışlıkla düşebilir.

Önerilen aralık:

- `0.45 - 0.6`

Debug'da bak:

- `[ref-player-mask-conflict] ... iou=...`
- IoU düşükken player düşüyorsa eşik düşük veya diğer şartlar fazla gevşektir.

### `--ref-player-conflict-mask-fill`

Varsayılan: `0.55`

Player bbox'ı içinde dominant mask'in ne kadar alan kaplaması gerektiğini belirler. Bu şart sağlanmadan player/referee conflict suppression uygulanmaz.

Artırınca:

- Sadece çok güçlü mask çakışmalarında player düşer.
- Yanlış player drop azalır.

Azaltınca:

- Daha küçük mask örtüşmeleri conflict sayılır.
- Mask drift varsa risklidir.

Önerilen aralık:

- `0.55 - 0.75`

Debug'da bak:

- `[ref-player-mask-conflict] ... fill=...`

### `--ref-player-conflict-conf-margin`

Varsayılan: `0.05`

Referee confidence değerinin player confidence değerinden ne kadar yüksek olması gerektiğini belirler.

Artırınca:

- Referee ancak belirgin şekilde daha güvenliyse player düşer.
- Player yanlış drop azalır.

Azaltınca:

- Referee/player temizliği daha agresif olur.

Önerilen aralık:

- `0.05 - 0.15`

Debug'da bak:

- Mevcut log confidence değerlerini conflict satırında göstermiyor. Gerekirse eklenebilir.

## Detection NMS ve Track NMS

Bunlar CLI'da şu an expose edilmedi, config tarafında var.

### `detection_nms_iou`

Varsayılan: `0.65`

Tracker'a girmeden önce aynı role ait yakın detectionları temizler.

Artırınca:

- Daha fazla duplicate detection kalır.
- Duplicate track riski artabilir.

Azaltınca:

- Duplicate detection daha agresif silinir.
- Yakın oyuncularda gerçek kutu düşebilir.

### `cross_role_iou`

Varsayılan: `0.80`

Player/referee detectionları birbirine çok çakışıyorsa düşük güvenli olanın temizlenmesinde kullanılır.

Artırınca:

- Player/referee aynı bölgede birlikte kalabilir.
- Çakışma temizliği daha az agresif olur.

Azaltınca:

- Ref/player role conflict daha agresif temizlenir.
- Yakın oyuncu/referee durumlarında gerçek detection düşebilir.

### `track_nms_iou`

Varsayılan: `0.88`

Tracker çıkışından sonra aynı role ait çok çakışan trackleri temizler.

Artırınca:

- Duplicate track daha fazla kalabilir.

Azaltınca:

- Duplicate track daha agresif düşer.
- Yakın oyuncularda yanlış track drop riski artar.

## Debug Logları Nasıl Okunur?

### `[det-debug]`

Örnek:

```text
[det-debug] frame=282 src=6276 raw_players=11 raw_refs=3 raw_numbers=6 filtered_players=9 filtered_refs=3 dropped_players=2 kp=12/18
```

Anlamı:

- `raw_players`: RF-DETR player detection sayısı.
- `filtered_players`: court/keypoint ve NMS sonrası tracker'a verilen player sayısı.
- `dropped_players`: tracker'a verilmeden düşen player sayısı.
- `kp`: court keypoint sayısı.

`dropped_players` sık artıyorsa problem MCByte'tan önce olabilir.

### `[mcbyte-assoc:player]`

Örnek:

```text
[mcbyte-assoc:player] frame=293 input=9 high=8 low=1 score_range=(0.567,0.937)
```

Anlamı:

- `input`: MCByte'a giren detection sayısı.
- `high`: `track_thresh` üstü detection sayısı.
- `low`: `0.1 < score < track_thresh` detection sayısı.
- `score_range`: frame içindeki detection score aralığı.

Step satırları:

```text
step=1 pool=11 dets=8 matches=8 unmatched_tracks=3 unmatched_dets=0 thresh=0.80 min=0.088 matched_max=0.538
```

Anlamı:

- `pool`: eşleşmeye giren track sayısı.
- `dets`: eşleşmeye giren detection sayısı.
- `matches`: başarılı eşleşme sayısı.
- `unmatched_tracks`: eşleşemeyen track sayısı.
- `unmatched_dets`: eşleşemeyen detection sayısı.
- `min`: cost matrix içindeki en iyi cost.
- `matched_max`: eşleşenler arasındaki en kötü cost.

### `[id-new]`, `[id-lost]`

ID'nin nerede başladığını/kaybolduğunu gösterir. Çok sık dönüşüyorsa ya detection kaybı, ya suppression, ya da association kırılması vardır.

### `[mask-duplicate]`

Aynı mask'i paylaşan tracklerden biri düşürüldü demektir.

Çok sık görünüyorsa:

- Mask drift ediyor olabilir.
- `mask_duplicate_min_fill` fazla düşük olabilir.
- Duplicate suppression mantığı gerçek oyuncuyu düşürüyor olabilir.

### `[ref-player-mask-conflict]`

Player track referee conflict nedeniyle düşürüldü demektir.

Çok sık görünüyorsa:

- Ref/player suppression fazla agresif olabilir.
- Referee detection player'a biniyor olabilir.
- Mask drift player'ı referee maskesine taşıyor olabilir.

## Başlangıç Deneme Komutları

Dengeli başlangıç:

```bash
python -m APP \
  --input videos/input/nba_game_h264.mp4 \
  --output videos/output/DEBUG.mp4 \
  --log-file videos/output/LOG.log \
  --max-frames 300 \
  --start 200 \
  --debug \
  --confidence 0.4 \
  --track-thresh 0.6 \
  --new-track-thresh 0.7 \
  --assoc1-thresh 0.8 \
  --assoc2-thresh 0.5 \
  --cmc-method orb
```

Mask suppression daha güvenli test:

```bash
--mask-duplicate-min-fill 0.6 \
--ref-player-conflict-iou 0.6 \
--ref-player-conflict-mask-fill 0.7 \
--ref-player-conflict-conf-margin 0.1
```

CMC etkisini izole etmek:

```bash
--cmc-method none
```

Track kopuyorsa:

```bash
--track-thresh 0.55 \
--assoc1-thresh 0.85
```

Yanlış match varsa:

```bash
--track-thresh 0.65 \
--assoc1-thresh 0.75
```

## Pratik Karar Ağacı

Çok ID kaybı varsa:

- Önce `[det-debug] dropped_players` kontrol et.
- Dropped yüksekse MCByte değil, pre-filter tarafına bak.
- Dropped düşük ama `unmatched_tracks` yüksekse association parametrelerine bak.

Çok yeni ID açılıyorsa:

- `new_track_thresh` artır.
- `unconfirmed_assoc_thresh` azaltmayı dene.
- `[mcbyte-assoc] init accepted` satırını kontrol et.

Mask yüzünden track düşüyorsa:

- `mask_duplicate_min_fill` artır.
- `ref_player_conflict_*` eşiklerini sıkılaştır.
- Gerekirse mask suppression'ı config ile kapatılabilir hale getirmek iyi bir sonraki adımdır.

Yakın oyuncularda swap varsa:

- `assoc1_thresh` çok düşükse track kopar, çok yüksekse yanlış match olabilir.
- `0.75 - 0.85` aralığında küçük adımlarla dene.
- CMC için `orb` ve `none` karşılaştır.

