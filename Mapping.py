import cv2
import os
import numpy as np
from time import sleep
upperfolder = r"C:\Users\Pc\Desktop\Yolo Kodları\Otonom görev\Test\Mapping"
proccessing_folder = r"C:\Users\Pc\Desktop\Yolo Kodları\Otonom görev\Test\Mapping\Current_map"
generalfolder = r"C:\Users\Pc\Desktop\Yolo Kodları\Otonom görev\Test\Mapping\Pictures" 
os.chdir(proccessing_folder)

last_photo = 2 #ba

def haritala(): 
    fotoğraflar = sorted(os.listdir(proccessing_folder))  # sıralı olsun
    sift = cv2.SIFT_create()
    anahtarnoktalar = []
    açıklamalar = []
    for foto in fotoğraflar:
        print(foto)
        img = cv2.imread(foto)
        #heigt, width = img.shape[:2]
        #cx, cy = width/2, heigt/2
        #fx = 4637.0
        #fy = 4637.0
        #K = np.array([
        #    [fx, 0.0,    float(cx)],
        #    [0.0,    fy, float(cy)],
        #   [0.0,    0.0,    1.0]
        #])
        #D = np.array([-0.1, 0.01, 0.0, 0.0, 0.0])
        #undistorted_img = cv2.undistort(img, K, D)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #(convertcolor) fotoğrafı gri tonlamasındaki bir formata dönüştürür.
        kps, desc = sift.detectAndCompute(gray, None) #kps = kp'lerin x ve y deki kordinatları, desc kp'lerin kimliği ve ne olduğunu söyler.
        anahtarnoktalar.append(kps)
        açıklamalar.append(desc)

    BFMatcher = cv2.BFMatcher() #bu kps'lerin desc'lerine bakarak fotoğrafları eşleştirir.
    eşleşenler = []
    matches = BFMatcher.knnMatch(açıklamalar[0], açıklamalar[1], k=2) #birinci fotoğraftaki(açıklamalar[0]) her tanımlayıcı için
                                                                      #ikinci fotoğraftaki en iyi 2(k) eşleşmeyi bul.
    iyiler = []
    for m, n in matches:
        if m.distance < n.distance * 0.75:
            iyiler.append(m)
    eşleşenler.append(iyiler)
    harita = cv2.imread(fotoğraflar[0])#ilk fotoğrafı referans alır.
    for i in range(len(eşleşenler)):
        foto1 = cv2.imread(fotoğraflar[i])
        foto2 = cv2.imread(fotoğraflar[i + 1])
        nokta1 = anahtarnoktalar[i]
        nokta2 = anahtarnoktalar[i + 1]
        iyi = eşleşenler[i]

        # for ortak_nokta in anahtarnoktalar[i]:
        #     print(ortak_nokta)

        debug_img = cv2.drawMatches(foto1, nokta1, foto2, nokta2, iyi, None, flags=2)
        debug_img = cv2.resize(debug_img, (1000, 500)) # Resize to fit screen

        if len(iyi) < 10:
            print(f"{i+1}. fotoğraf için yeterli eşleşme yok, geçildi.")
            continue

        #altaki iki kod satırı "iyiler" listesini alır ve bu eşleşmelere karşılık gelen noktaların (x, y) koordinatlarını iki ayrı liste halinde düzenler:
        src_pts = np.float32([nokta1[m.queryIdx].pt for m in iyi]).reshape(-1, 1, 2) #birinci fotoğraftaki eşleşmeler(source = kaynak)
        dst_pts = np.float32([nokta2[m.trainIdx].pt for m in iyi]).reshape(-1, 1, 2) #ikinci fotoğraftaki eşleşmeler(destination = hedef)
        #.pt = x ve y konumu, ...Idx fotoğraftaki kps'lerin listesindeki sıra numarasını verir.
        #reshape():
        #   2: Verileri 2'şerli grupla (x ve y).
        #   1: Her 2'li grubu kendi 1 satırlık matrisine koy(listenin içerisine iki tane eleman).
        #   -1: Geri kalan satır sayısını sen otomatik olarak hesapla.
        #bu kod findHomography() komutun çalışması için ona uygun komutları verir.

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 10.0)
        #h = homografi: ikinci fotoğrafı birinci fotoğrafın üstüne oturttuğumuzda mükemmel gözükmesi için nasıl döndürmemiz ve haraket ettirmemizin hesaplamasıdır.
        #mask = eşleşen noktaların güvenli, iyi veya hatalı olup olmadığını söyler.
        #iyiler listesindekiler hala hatalı olabilir. bunları engeller.
        #10.0 = h matrisini kullanarak noktayı dönüştürdüğümüzde gerçek uzaklıktan en fazla ne kadar uzakta olabilir? eğer 10'dan fazla ise oylamaya dahil etmez

        h1, w1 = foto1.shape[:2]
        h2, w2 = foto2.shape[:2]
        corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2) #ikinci fotoğrafın köşelerinin kordinatlarını liste içine alır.
        corners_img2_trans = cv2.perspectiveTransform(corners_img2, H) #corners_img2 listesini alır ve bu köşelere H matrisini uygular.
        corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2) #birinci fotoğrafın köşelerinin kordinatlarını liste içine alır.
        #corners_img2 = [[[x1, y1]],#[[x2, y2]],#[[x3, y3]],[[x4, y4]]]
        #corners_img2_trans = [[[x5, y5]],#[[x6, y6]],#[[x7, y7]],[[x8, y8]]]  

        all_corners = np.concatenate((corners_img1, corners_img2_trans), axis=0) #axis=0 alt alta birleştir, axis = 1 yan yana birleştirir.
        #all_corners alt alta corners_img1 ve corners_img2_trans dizelerini birleştirir.
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5) #satırlar boyunca en küçük x ve y değerlerini bulur.
        #ravel() = içine yazılan çok boyutlu dizeyi tek boyutlu hale getirir.
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5) #satırlar boyunca en büyük x ve y değerlerini bulur.

        #x_min: Haritanın en sol noktası.
        #y_min: Haritanın en üst noktası.
        #x_max: Haritanın en sağ noktası.
        #y_max: Haritanın en alt noktası.

        # Ofset (negatif koordinatlar için)
        translation = np.array([[1, 0, -x_min],  #haritanın sol üst küşesini (0,0) kordinatına ayarlar.
                                [0, 1, -y_min],
                                [0, 0, 1]])
        # Yeni canvas boyutu
        canvas_w = x_max - x_min
        canvas_h = y_max - y_min
        # Yeni canvas oluştur ve fotoğrafları yerleştir
        canvas = cv2.warpPerspective(foto2, translation @ H, (canvas_w, canvas_h))
        #translation @ H: Python, "Önce H'yi uygula, sonra translation'ı uygula" talimatlarını tek bir talimatta birleştirir.
        #(canvas_w, canvas_h) canvasın yüksekliğini ve genişliğini belirler.

        ##############################################################################################
        y_offset = abs(y_min)
        x_offset = abs(x_min)

        # 1. Create a "Mask" of the Old Map (Find all pixels that are NOT black)
        # Convert to gray to check brightness
        tmp_gray = cv2.cvtColor(foto1, cv2.COLOR_BGR2GRAY)
        # Create a mask: 255 for valid pixels, 0 for black pixels
        _, mask = cv2.threshold(tmp_gray, 1, 255, cv2.THRESH_BINARY)

        # 2. Define the Region of Interest (ROI) on the canvas where the Old Map goes
        roi = canvas[y_offset:y_offset+h1, x_offset:x_offset+w1]

        # 3. Paste ONLY the valid parts of foto1 using the mask
        # "Where the mask is white, use foto1's pixel. Otherwise, keep the canvas pixel (New Photo)."
        roi[mask == 255] = foto1[mask == 255]
        
        # Update the canvas with the blended ROI
        canvas[y_offset:y_offset+h1, x_offset:x_offset+w1] = roi
        ##############################################################################################
        
        harita = canvas
        print(f"harita oluşturuldu!")
    return harita

while True:
    harita = haritala()
    last_photo += 1
    os.remove("1.jpg")
    os.remove("2.jpg")
    cv2.imwrite("1.jpg", harita)
    os.chdir(upperfolder)
    new_src = fr"Pictures/{last_photo}.jpg"
    new_dst = fr"Current_map/{2}.jpg"
    os.rename(new_src,new_dst)
    os.chdir(proccessing_folder)
    if len(os.listdir(generalfolder)) < 3:
        print("#########################")
        break
    print(last_photo)
    sleep(3)

#harita = haritala()
#cv2.imwrite("Harita.jpg", harita)
