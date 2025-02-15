import re
import webbrowser
import googleapiclient.discovery
from sentence_transformers import SentenceTransformer, util

# all-mpnet-base-v2 比較歌曲標題與使用者輸入的相似度 基本上比較偏向要直接輸入歌名或者歌手了
model = SentenceTransformer("all-mpnet-base-v2")

# YouTube API金鑰  差點要在GITHUB上公布了
YOUTUBE_API_KEY = "" #這我不能給吧? 用自己的去

# 我的YOUTUBE歌單
YOUTUBE_PLAYLIST_ID = "PLXuNaCgnseuvZPDI1DfyMXSSsbtF4KHL7"
YOUTUBE_BASE_URL = "https://www.youtube.com/watch?v="


def get_playlist_videos(api_key, playlist_id):
    #取得指定歌單中的所有歌曲資訊 
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    
    songs = []
    next_page_token = None

    while True:
        request = youtube.playlistItems().list(
            part="snippet",#搜尋標題
            playlistId=playlist_id,
            maxResults=50, #一次最多取得 50 筆資料。
            pageToken=next_page_token#還不夠就往下一頁尋找
        )
        response = request.execute()
        
        for item in response["items"]:
            video_title = item["snippet"]["title"] #取得影片標題
            video_id = item["snippet"]["resourceId"]["videoId"] #取得影片唯一識別碼
            video_link = f"{YOUTUBE_BASE_URL}{video_id}" #取得歌曲url
            songs.append({"title": video_title, "link": video_link, "raw_title": video_title})
        
        next_page_token = response.get("nextPageToken")#如果有下一頁，則繼續請求 API
        if not next_page_token:
            break
    
    return songs

def find_best_matches(user_query, song_list, top_n=5):
    
    user_embedding = model.encode(user_query, convert_to_tensor=True) #將使用者輸入轉換成詞向量
    song_titles = [song["title"] for song in song_list]#將所有歌曲標題轉換成詞向量
    song_embeddings = model.encode(song_titles, convert_to_tensor=True)
    
    # 計算相似度（cosine similarity）
    similarities = util.pytorch_cos_sim(user_embedding, song_embeddings)[0] #計算語意相似度
    
    # 取出Top-N相似度最高的歌曲
    top_indices = similarities.argsort(descending=True)[:top_n]
    
    return [song_list[idx] for idx in top_indices]

def play_song(url):
    #預設瀏覽器開始播放歌曲
    webbrowser.open(url)

if __name__ == "__main__":
    print("正在獲取歌單歌曲...")
    songs = get_playlist_videos(YOUTUBE_API_KEY, YOUTUBE_PLAYLIST_ID)
    
    while True:
        user_input = input("\n輸入想聽的歌曲描述，輸入exit離開：")
        if user_input.lower() == "exit":
            break
        
        best_songs = find_best_matches(user_input, songs, top_n=10)  #推薦前10 首
        print("\n最符合的歌曲：")
        for i, song in enumerate(best_songs):
            print(f"{i+1}. {song['raw_title']} - {song['link']}")
        
        # 讓用戶選擇要播放哪首
        choice = input("\n請選擇要播放的歌曲編號(1-10)，或者按其他的取消選擇：")
        if choice.isdigit() and 1 <= int(choice) <= 10:
            play_song(best_songs[int(choice) - 1]["link"])
