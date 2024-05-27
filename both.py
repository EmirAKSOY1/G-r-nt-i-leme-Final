from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import yt_dlp #youtubedan video indirmemizi sağlayan kütüphane

url = 'https://www.youtube.com/watch?v=DpmRVgiYcII' #Canlı videonun olduğu youtube url


def get_stream_url(url):# Url olarak verilen videoyu indirmey yarayan fonksiyon
    ydl_opts = {
        'format': 'best', # Videoyu yüksek çözünürlükte indirir
        'quiet': True,  # Terminale herhangi bir şey yazdırmaz
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: # yukarıda verilen parametrelerle videoyu çeker
        info = ydl.extract_info(url, download=False) #videonun bilgilerini alır (videonun başlığını ,fps,çözünürlük,boyutları vs..)
        print(info['title'])#videonun başlığını verir
        print(info['fps'])#videonun fps sini verir
        return info['url']#bilgileri döndürür

model = YOLO("yolov8n.pt") # YOLO V8 nesne takip modelini yükler
print(model.names)#modeldeki sınıfların isimleri 80 adet sınıf bulunuyor
stream_url = get_stream_url(url)
"""
stream_url'in değeri:
https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1716843156/ei/NJ5UZvXlA5C_6dsP956K8AM/ip/88.225.253.168/id/DpmRVgiYcII.1/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D137/rqh/1/hdlc/1/hls_chunk_host/rr1---sn-u0g3uxax3-txpl.googlevideo.com/xpc/EgVo2aDSNQ%3D%3D/playlist_duration/30/manifest_duration/30/vprv/1/playlist_type/DVR/initcwndbps/871250/mh/ZC/mm/44/mn/sn-u0g3uxax3-txpl/ms/lva/mv/m/mvi/1/pl/22/dover/11/pacing/0/keepalive/yes/mt/1716821075/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,rqh,hdlc,xpc,playlist_duration,manifest_duration,vprv,playlist_type/sig/AJfQdSswRQIgYUbjGTWWE24EeQBND2dclzTAecPZp2PfagrAcWjwkKACIQCKFL4XkWSwxxPVMkN6ZPSVjwFl-EX8C26rjnX9zKNy7w%3D%3D/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pl/lsig/AHWaYeowRgIhAPbmN_g-Es8L2hLAJB838oDOtGrBbdcg8vmGSk9DfaeyAiEAwwhXwmbmlKt00cAvzTc0tlloR4SRGTTsD7DC2q9J1Io%3D/playlist/index.m3u8
yani aslında VideoCapture(stream_url) da stream_url yerine yukarıdaki url i de yazsak yine çalışacak.
Fonksiyonun amacı aslında videonun manifest url ini almak . Manifest url i de videonun gerekli segmentlerini ve akış bilgilerini veriyor.

"""

cap = cv2.VideoCapture(stream_url)#manifest url'i ile opencv ye videonun segmentlerini gönderiyoruz
noktalar = [(580, 150), (1100, 150), (1100, 250), (580, 250)]#hesaplama yapılacak bölgenin koordinatları 
en= 1280
boy= 720 #video çok geniş olduğu için bu şekilde bir ölçeklendirme yaptım ekrana sığması için

# Videoyu kaydetmeye yarayan fonksiyon
video_writer = cv2.VideoWriter("object_counting_output.mp4", #videonun ismi
                               cv2.VideoWriter_fourcc(*'mp4v'),#videonun formatı 
                               cv2.CAP_PROP_FPS,#youtub videosunun fpsine göre kaydedyoruz
                               (en, boy))#video çok geniş olduğu için ölçülerini ben kendim belirledim

counter = object_counter.ObjectCounter() #Asıl işlemi yapan kısım burası sayacı oluşturuyoruz
#alt satırda da sayacın parametrelerini belirliyorum
counter.set_args(view_img=True,#işlenen karelerin görüntülerini ekranda verir
                 reg_pts=noktalar,#Hesaplaacak bölgenin koordinatları
                 classes_names=model.names,#modeldeki sınıfların isimleri
                 draw_tracks=True,#takip edilen nesnelerin yolunu çizer , false verilirse sistem çalışmaz çünkü nesnenin yolu belli olmaz
                 line_thickness=2) #çizgi kalınlığı

while cap.isOpened():#videoyu döngüye sokarak kareleri işleyeceğiz
    success, video = cap.read()#youtubedan gelen canlı videoyu okuyoruz eğer video sorunsuz çalışırsa success true olur
    if not success: #video başarızsa döngüden çık
        break

    # kareleri verdiğim ölçüye göre yeniden boyutlandırıyorum
    re_video = cv2.resize(video, (en, boy))

    tracks = model.track(re_video, persist=True, show=False)#bu fonksiyon kişilerin hareketlerini takip eder ilk parametre video kaynağı , ikinci 
    #parametre nesnelerin geçmiş konumlarını tutarak kişilerin giriş mi çıkış mı yatığını anlar ,üçüncü parametre tespit ettiği nesnelerin skorlarını yani doğrulunu da yazdırır etikete 
    #üçüncü parametreye true verdiğmde nerdeyse 1 fps e düşüyor

    im0_resized = counter.start_counting(re_video, tracks)#counterın parametreleiriniyukarda vermiştim burada saymayı gerçekleştiriyor
    video_writer.write(re_video)#vieoyu yazar

    # ekrn çıktısı
    cv2.imshow("Emir Aksoy", re_video)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc tuşu ile çıkış
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()#tüm pencereleri kapatır
