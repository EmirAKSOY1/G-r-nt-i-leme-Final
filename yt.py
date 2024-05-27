import cv2
import yt_dlp

# YouTube video URL'si
url = 'https://www.youtube.com/watch?v=DpmRVgiYcII'

# Video akış URL'sini almak için yt-dlp kullanma
def get_stream_url(url):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        print(info['title'])
        return info['url']

stream_url = get_stream_url(url)
print(stream_url)
# OpenCV ile canlı video akışını işleme
cap = cv2.VideoCapture(stream_url)
while True:
    ret, frame = cap.read()
    # Gri tonlamalı frame'i göster
    cv2.imshow('Gray Live Stream', frame)
    
    # 'q' tuşuna basarak çıkış yapın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
