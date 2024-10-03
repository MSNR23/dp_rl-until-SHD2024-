import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

# CSVファイルからデータを読み取る関数
def read_csv_data(csv_file_path):
    time_data = []
    theta1_data = []
    theta2_data = []

    with open(csv_file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # ヘッダー行をスキップ
        for row in csv_reader:
            time_data.append(float(row[0]))  # Timeデータを読み取る
            theta1_data.append(float(row[1])*np.pi / 180)  # Theta1データを読み取る
            theta2_data.append(float(row[2])*np.pi / 180)  # Theta2データを読み取る

    return time_data, theta1_data, theta2_data

# アニメーションの初期化関数
def init():
    pendulum1.set_data([], [])
    pendulum2.set_data([], [])
    trajectory.set_data([], [])  # 軌跡の初期化
    return pendulum1, pendulum2, trajectory

# アニメーションの更新
def update(frame, theta1_data, theta2_data, x2_traj, y2_traj):
    x1 = np.cos(theta1_data[frame])  # 振り子1のx座標
    y1 = np.sin(theta1_data[frame])  # 振り子1のy座標
    pendulum1.set_data([0, x1], [0, y1])  # 振り子1の位置を更新

    x2 = x1 + np.cos(theta1_data[frame] + theta2_data[frame])  # 振り子2のx座標
    y2 = y1 + np.sin(theta1_data[frame] + theta2_data[frame])  # 振り子2のy座標
    pendulum2.set_data([x1, x2], [y1, y2])  # 振り子2の位置を更新

    # 軌跡を記録してリアルタイムで表示
    x2_traj.append(x2)
    y2_traj.append(y2)
    trajectory.set_data(x2_traj, y2_traj[:len(x2_traj)])  # 軌跡の更新

    return pendulum1, pendulum2, trajectory

# CSVファイルのパス
csv_file_path = 'try_2070.csv'

# CSVファイルからデータを読み取る
time_data, theta1_data, theta2_data = read_csv_data(csv_file_path)

# 軌跡を保存するためのリスト
x2_traj = []
y2_traj = []

# アニメーションの設定
fig, ax = plt.subplots()
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
pendulum1, = ax.plot([], [], lw=2, label='Link 1')
pendulum2, = ax.plot([], [], lw=2, label='Link 2')
trajectory, = ax.plot([], [], 'r-', lw=1, label='Trajectory')  # 軌跡の追加
ax.legend()

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=len(time_data), fargs=(theta1_data, theta2_data, x2_traj, y2_traj), init_func=init, blit=False, interval=1)

# アニメーションをMP4形式で保存
ani.save('0926_1.900.14_10.mp4', writer='ffmpeg', fps=10)
plt.show()
