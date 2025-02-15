import re
import webbrowser
import googleapiclient.discovery
import tensorflow as tf
from transformers import CLIPTokenizer, TFCLIPModel
import numpy as np

# 使用clip，偏向語意搜尋，但是搜索結果是真的很差
# 這很偏向於是靠著描述一段歌曲介紹來去尋找我想要的歌曲，但是這絕對會需要對於每個想要找的歌曲進行更深入的消息探索並進行微調才能做到
clip_model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# 設定 YouTube API
YOUTUBE_API_KEY = "" #這我不能給吧? 用自己的去
YOUTUBE_PLAYLIST_ID = "PLXuNaCgnseuvZPDI1DfyMXSSsbtF4KHL7"
YOUTUBE_BASE_URL = "https://www.youtube.com/watch?v="

def clean_title(title): #處理歌曲標題
    
    title = re.sub(r"\(.*?\)|\[.*?\]|\{.*?\}", "", title)  # 移除括號內內容
    title = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5 ]", "", title)  # 移除特殊字符
    return title.strip().lower()

def get_playlist_videos(api_key, playlist_id):
    #取得 YouTube 歌單中的所有歌曲資訊
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    
    songs = []
    next_page_token = None

    while True:
        request = youtube.playlistItems().list(
            part="snippet", #搜尋標題
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()
        
        for item in response["items"]:
            video_title = item["snippet"]["title"]
            video_id = item["snippet"]["resourceId"]["videoId"]
            video_link = f"{YOUTUBE_BASE_URL}{video_id}"
            cleaned_title = clean_title(video_title)  #標題處理 最終留下想要的歌曲名稱以及歌手 把特殊符號去除
            songs.append({"title": cleaned_title, "link": video_link, "raw_title": video_title})
        
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    
    return songs

def encode_text_clip(text):
    #CLIP將文本轉為向量
    inputs = clip_tokenizer([text], return_tensors="tf", padding=True, truncation=True)
    text_features = clip_model.get_text_features(**inputs)
    return tf.nn.l2_normalize(text_features, axis=-1)  # L2 正規化

def find_best_matches(user_query, song_list, top_n=5):
    # 轉換使用者輸入為向量
    user_embedding = encode_text_clip(user_query)
    # 轉換所有歌曲標題為向量
    song_titles = [song["title"] for song in song_list]
    song_embeddings = tf.concat([encode_text_clip(title) for title in song_titles], axis=0)
    # 使用 tf.tensordot 計算 cosine similarity
    similarities = tf.tensordot(user_embedding, song_embeddings, axes=[[1], [1]])
    # 轉換為NumPy陣列，確保是向量
    similarities = similarities.numpy().flatten()
    # 取出Top-N相似度最高的索引
    top_indices = np.argsort(-similarities)[:top_n]  #使用負號排序（從高到低）

    return [song_list[idx] for idx in top_indices]

def play_song(url):
    """用預設瀏覽器播放歌曲的URL"""
    webbrowser.open(url)

if __name__ == "__main__":
    print("歌單歌曲 尋找中")
    songs = get_playlist_videos(YOUTUBE_API_KEY, YOUTUBE_PLAYLIST_ID)
    
    while True:
        user_input = input("\n請輸入你想聽的歌曲描述（或輸入 'exit' 離開）：")
        if user_input.lower() == "exit":
            break
        
        best_songs = find_best_matches(user_input, songs, top_n=5)  #前五個推薦的歌曲
        print("\n🔍 最符合的歌曲：")
        for i, song in enumerate(best_songs):
            print(f"{i+1}. {song['raw_title']} - {song['link']}")
        
        #選擇要播放哪首
        choice = input("想聽的歌曲（輸入 1-5），也可以按其他的跳過：")
        if choice.isdigit() and 1 <= int(choice) <= 5:
            play_song(best_songs[int(choice) - 1]["link"])
