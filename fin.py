import pandas as pd
import matplotlib.pyplot as plt
import glob
import re

# Qテーブルが保存されているディレクトリのパス
csv_files_path = './1.90_0.14_finQ/*.csv'

# 全てのエピソードごとのCSVファイルを読み込む
csv_files = glob.glob(csv_files_path)

# 各エピソードの最大報酬を保存するリスト
max_rewards = []

# 各エピソード番号を保存するリスト
episodes = []

# CSVファイルごとに処理を行う
for file in csv_files:
    # ファイル名からエピソード番号を抽出
    match = re.search(r'episode_(\d+)', file)
    if match:
        episode_number = int(match.group(1))
        
        # Qテーブルを読み込む
        q_table = pd.read_csv(file)
        
        # 'value'列が存在するか確認して最大値を取得
        if 'value' in q_table.columns:
            # 'value'列を強制的に数値型に変換し、無効な値を除去
            q_table['value'] = pd.to_numeric(q_table['value'], errors='coerce')
            max_reward = q_table['value'].dropna().max()
            max_rewards.append(max_reward)
            episodes.append(episode_number)
            
            # デバッグ出力で確認
            print(f"ファイル: {file}, エピソード: {episode_number}, 最大値: {max_reward}")
        else:
            print(f"'value'列が見つかりません: {file}")
            continue  # 列がない場合は次へ

# エピソード番号でソート
sorted_data = sorted(zip(episodes, max_rewards))

# ソート後のエピソードと報酬リストに分解
episodes, max_rewards = zip(*sorted_data)

# データを保存するDataFrameを作成
df = pd.DataFrame({
    
    'Episode': episodes,
    'Max_Reward': max_rewards                          
})

# 結果をCSVファイルに保存
df.to_csv('1.90_0.14_max_rewards.csv', index=False)

# グラフの描画
plt.figure(figsize=(10, 6))
plt.plot(episodes, max_rewards, marker='o', linestyle='-', color='b')
plt.title('Maximum Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Max Reward')
plt.grid(True)
plt.show()
