import cv2
import sys


def play_video(file_path):
    # 動画ファイルを開く
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print(f"Error: Cannot open the file {file_path}")
        sys.exit(1)

    # 動画再生ループ
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Video playback finished.")
            break

        # フレームを表示
        cv2.imshow("Video", frame)

        # 'q'キーが押されたら終了
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Video playback interrupted.")
            break

    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python play_video.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    play_video(video_file)
