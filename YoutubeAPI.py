import os
import requests
import yt_dlp

# YouTube Data API configuration
API_KEY = "AIzaSyAAIbkEV_mdghy9OahxwH_RhXQwhw6vSeY"
SEARCH_QUERY = "Carnatic vocal solo"
MAX_RESULTS = 40  # Number of videos to fetch
OUTPUT_PATH = "downloads"

def search_youtube_videos(query, max_results=1):
    """
    Searches YouTube for videos based on the query and returns a list of video URLs.

    Args:
        query (str): Search term.
        max_results (int): Maximum number of video results.

    Returns:
        list: A list of video URLs.
    """
    base_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": API_KEY,
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        videos = response.json().get("items", [])
        video_urls = [
            f"https://www.youtube.com/watch?v={video['id']['videoId']}" for video in videos
        ]
        return video_urls
    else:
        print("Error fetching videos:", response.text)
        return []

def download_audio_with_ytdlp(video_url, output_path="C:\\Users\\nandh\\Downloads\\Carnatic Music"):
    """
    Downloads audio from a YouTube video using yt-dlp.

    Args:
        video_url (str): URL of the YouTube video.
        output_path (str): Directory where the audio will be saved.
    """
    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # yt-dlp options for downloading audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'ffmpeg_location': r'C:\\ffmpeg\\bin',  # Replace with your actual ffmpeg path
        }

        print(f"Audio will be saved to: {os.path.join(output_path, '%(title)s.%(ext)s')}")

        # Download the audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading audio from: {video_url}")
            ydl.download([video_url])

    except Exception as e:
        print(f"Error downloading audio for {video_url}: {e}")


def main():
    # Step 1: Search for videos
    print(f"Searching for YouTube videos about '{SEARCH_QUERY}'...")
    video_urls = search_youtube_videos(SEARCH_QUERY, MAX_RESULTS)

    if not video_urls:
        print("No videos found.")
        return

    print(f"Found {len(video_urls)} videos. Starting download...")

    # Step 2: Download audio for each video
    for url in video_urls:
        download_audio_with_ytdlp(url, OUTPUT_PATH)

if __name__ == "__main__":
    main()
print("Output Path:", OUTPUT_PATH)
print("Files in Output Path:", os.listdir(OUTPUT_PATH))