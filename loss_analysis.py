from collections import defaultdict
import torch


def get_high_loss_videos(video_loss_dict, video_id_to_path, logits_labels):
    """動画ごとの損失を計算し、高い損失を持つ動画を返す"""
    sorted_loss_videos = sorted(video_loss_dict.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)[:5]

    # 高いlossの動画ファイル名をターミナルに出力
    # print("High loss videos (top 5):")
    # result = []
    # for video_id, losses in sorted_loss_videos:
    #    avg_loss = sum(losses) / len(losses)  # lossの平均
    #    video_path = video_id_to_path.get(video_id, "Unknown video path")
    #    result.append((video_id, avg_loss, video_path))
    #    print(f"Video ID: {video_id}, Loss: {avg_loss}, Video Path: {video_path}")

    # return result
    high_loss_videos = []
    for video_id, loss_list in sorted_loss_videos:
        video_path = video_id_to_path.get(video_id, "Unknown")
        avg_loss = sum(loss_list) / len(loss_list)

        # Fetch logits and labels for the video
        # logits, labels = logits_labels.get(video_id, (None, None))
        # 取得処理を変更
        records = logits_labels.get(video_id, [])
        if records:
            logits = [record["probabilities"] for record in records]  # 確率リストの取得
            labels = [record["label"] for record in records]          # ラベルリストの取得
        else:
            logits, labels = None, None

        # if logits is not None and labels is not None:
        #     predicted_label = logits.argmax(dim=-1).item()
        #     true_label = labels.item()
        #     is_correct = int(predicted_label == true_label)
        # else:
        #     is_correct = -1  # Unknown status if logits/labels are missing

        if logits and labels:  # データが存在する場合
            is_correct_list = []
            for predicted_probs, true_label in zip(logits, labels):
                # 最大確率のインデックスを予測ラベルとする
                predicted_label = torch.tensor(predicted_probs).argmax(dim=-1).item()
                is_correct = int(predicted_label == true_label)
                is_correct_list.append(is_correct)

        else:
            is_correct_list = [-1]  # データがない場合、-1 を記録

    high_loss_videos.append((video_path, avg_loss, is_correct))

    return high_loss_videos
