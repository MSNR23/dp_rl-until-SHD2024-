import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import time
import os

# 定数

L = 1.90 # 身長

ma = 70 # 全体質量  

g = 9.80  # 重力加速度 

# リンクの長さ
l1 = 0.186 * L
l2 = 0.254 * L

# 前腕の長さ
l_forearm = 0.575 * l2
# 手の長さ
l_hand = 0.425 * l2
# 前腕の質量
m_forearm = 0.016 * ma
# 手の質量
m_hand = 0.006 * ma


# 質点の質量（m_hand + 投擲物）
m1 = 0.028 * ma
m2 = m_forearm + m_hand + 0.14
# m3 = 0.14 # 投擲物の質量

# 重心までの長さ（m_hand + 投擲物）
lg1 = l1 / 2
lg2 = (m_forearm * (l_forearm / 2) + m_hand * (l_forearm + l_hand / 2) + 0.14 * l2) / (m_forearm + m_hand + 0.14)

# 上腕の慣性モーメント
I1 = m1 * l1**2 / 12
# I2 = ((m_forearm * l_forearm**2 + m_hand * l_hand**2)/3 + 0.14 * l2**2)
# 前腕の慣性モーメント（平行軸の定理）
I2_forearm = m_forearm * l_forearm**2 / 12 + m_forearm * lg1**2
# 手の慣性モーメント（平行軸の定理）
I2_hand = m_hand * l_hand**2 / 12 + m_hand * (l_forearm + lg2)**2
# 投擲物の慣性モーメント
I2_ball = 0.14 * l2**2
# 前腕リンク全体の慣性モーメント
I2 = I2_forearm + I2_hand + I2_ball

# 粘性係数
b1 = 0.05
b2 = 0.01

# 初期条件
q10 =  0
#q20 = 15 * np.pi / 180
q20 = 0
q1_dot0 = 0.0
q2_dot0 = 0.0

dt = 0.005

# CSVファイルの保存先ディレクトリ
save_dir1 = r'1.90_0.14_max_R_bow'
save_dir2 = r'1.90_0.14_try_bow'
save_dir3 = r'1.90_0.14_maxQ_bow'
save_dir4 = r'1.90_0.14_finQ_bow'

# ディレクトリが存在しない場合は作成
if not os.path.exists(save_dir1):
    os.makedirs(save_dir1)

if not os.path.exists(save_dir2):
    os.makedirs(save_dir2)

if not os.path.exists(save_dir3):
    os.makedirs(save_dir3)

if not os.path.exists(save_dir4):
    os.makedirs(save_dir4)

# 運動方程式
def update_world(q1, q2, q1_dot, q2_dot, tau, action):
    # 行動に基づくトルク[Nm]を設定
    tau = np.zeros((2, 1))
    if action == 0:
        tau = np.array([[18.0], [0.0]])
    elif action == 1:
        tau = np.array([[-18.0], [0.0]])
    elif action == 2:
        tau = np.array([[0.0], [11.0]])
    elif action == 3:
        tau = np.array([[0.0], [-11.0]])
    elif action == 4:
        tau = np.array([[18.0], [11.0]])
    elif action == 5:
        tau = np.array([[-18.0], [-11.0]])
    elif action == 6:
        tau = np.array([[18.0], [-11.0]])
    elif action == 7:
        tau = np.array([[-18.0], [11.0]])
    elif action == 8:
        tau = np.array([[0.0], [0.0]])

    # リンク2が可動範囲の限界に達した場合の外力
    if q2 <= 0:
        tau[1, 0] += 20.0  # 0度のとき、正の方向に5N
    elif q2 >= np.radians(150):
        tau[1, 0] += -20.0  # 145度のとき、負の方向に5N

    # 質量行列
    M_11 = m1*lg1**2 + I1 + m2*(l1**2 + lg2**2 + 2*l1*lg2*np.cos(q2)) + I2
    M_12 = m2 * (lg2**2 + l1 * lg2*np.cos(q2)) + I2
    M_21 = m2 * (lg2**2 + l1*lg2 * np.cos(q2)) + I2
    M_22 = m2 * lg2**2 + I2

    M = np.array([[M_11, M_12],
                  [M_21, M_22]])

    # コリオリ行列
    C_11 = -m2 * l1 * lg2 * np.sin(q2) * q2_dot * (2 * q1_dot + q2_dot)
    C_21 = m2 * l1 * lg2 * np.sin(q2) * q1_dot**2
    C = np.array([[C_11], [C_21]])

    # 重力ベクトル
    G_11 = m1 * g * lg1 * np.cos(q1) + m2 * g * (l1 * np.cos(q1) + lg2 * np.cos(q1 + q2))
    G_21 = m2 * g * lg2 * np.cos(q1 + q2)
    G = np.array([[G_11], [G_21]])

    # 粘性
    B_11 = b1 * q1_dot
    B_21 = b2 * q2_dot
    B = np.array([[B_11], [B_21]])

    # 逆行列
    M_inv = np.linalg.inv(M)

    q_ddot = M_inv.dot(tau - C - G - B)


    return np.array([q1_dot, q2_dot, q_ddot[0, 0], q_ddot[1, 0]])

# Runge-Kutta法
def runge_kutta(t, q1, q2, q1_dot, q2_dot, action, dt):
    tau = np.zeros((2, 1))

    k1 = dt * update_world(q1, q2, q1_dot, q2_dot, tau, action)
    k2 = dt * update_world(q1 + 0.5 * k1[0], q2 + 0.5 * k1[1], q1_dot + 0.5 * k1[2], q2_dot + 0.5 * k1[3], tau, action)
    k3 = dt * update_world(q1 + 0.5 * k2[0], q2 + 0.5 * k2[1], q1_dot + 0.5 * k2[2], q2_dot + 0.5 * k2[3], tau, action)
    k4 = dt * update_world(q1 + k3[0], q2 + k3[1], q1_dot + k3[2], q2_dot + k3[3], tau, action)

    q1_new = q1 + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
    q2_new = q2 + (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
    q1_dot_new = q1_dot + (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6
    q2_dot_new = q2_dot + (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6

    # リンク2の角度を0~145度に制限
    q2_new = np.clip(q2_new, 0, np.radians(150))

    return q1_new, q2_new, q1_dot_new, q2_dot_new

max_number_of_steps = 2000 # 最大ステップ数
num_episodes = 5000

# Q学習のパラメータ
alpha = 0.1  # 学習率
gamma = 0.9  # 割引率
# epsilon = 0.5 * (0.99 ** (episode + 1)) # ε-greedy法のε

# Qテーブルのbin数
num_q1_bins = 4
num_q2_bins = 4
num_q1_dot_bins = 4
num_q2_dot_bins = 4
num_actions = 9  # 行動数

Q = np.zeros((num_q1_bins, num_q2_bins, num_q1_dot_bins, num_q2_dot_bins, num_actions))
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]
    
# 状態の離散化関数
def digitize_state(q1, q2, q1_dot, q2_dot):
    digitized = [np.digitize(q1, bins = bins(-np.pi, np.pi, num_q1_bins)),
                 np.digitize(q2, bins = bins(0, 150 * np.pi / 180, num_q2_bins)),
                 np.digitize(q1_dot, bins = bins(-10.0, 10.0, num_q1_dot_bins)),
                 np.digitize(q2_dot, bins = bins(-10.0, 10.0, num_q2_dot_bins))]

    return digitized[0], digitized[1], digitized[2], digitized[3]

# リセット関数
def reset():
    q1 = q10
    q2 = q20
    q1_dot = q1_dot0
    q2_dot = q2_dot0

    return q1, q2, q1_dot, q2_dot

# ε-greedy法に基づく行動の選択
def get_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, episode):
    epsilon = 0.5 * (0.99 ** (episode + 1)) # ε-greedy法のε
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, :])

    # return reward
def compute_reward(q1, q2, q1_dot, q2_dot, cumulative_energy):
    # 前腕リンクの手先速度を計算
    v_x2 = -l1 * np.sin(q1) * q1_dot - l2 * np.sin(q1 + q2) * (q1_dot + q2_dot)
    v_y2 = l1 * np.cos(q1) * q1_dot + l2 * np.cos(q1 + q2) * (q1_dot + q2_dot)
    v2 = np.sqrt(v_x2**2 + v_y2**2)

    # 前腕リンクの角度が45度に近いかを評価
    q1_deg = np.degrees(q1)
    # q2_deg = np.degrees(q2)

    # θvを定義
    # qv_deg = (q1_deg + q2_deg) - 90
    qv = np.arctan2(v_x2, v_y2)

    # θvを0〜360度の範囲に変換
    # qv_deg = qv_deg % 360  # 360度で剰余をとる
    qv_deg = np.degrees(qv) % 360  # 360度で剰余をとる

    # # qv_degが負の時は360を足して正の角度に変換
    # if qv_deg < 0:
    #     qv_deg += 360

    if 0 <= qv_deg <= 45:
        angle_reward = qv_deg / 45
    elif 45 < qv_deg <= 90:
        angle_reward = (90 - qv_deg) / 45
    # elif 90 < qv_deg <= 135:
    #     angle_reward = (qv_deg - 90) / 45
    # elif 135 < qv_deg <= 180:
    #     angle_reward = (180 - qv_deg) / 45
    else:
        angle_reward = 0

    reward = angle_reward * v2

    # 累積消費エネルギーをペナルティとして使用
    reward -= 0.001 * cumulative_energy  # 累積消費エネルギーによるペナルティ

    # if q1_deg > 0:
    #     reward += -10
    
    if q1_dot > 0:
        reward += -2

    if v_y2 < 0:
        reward += -1

    if q1_deg >  360 or q1_deg < -1000:
        reward += -15
    return reward

def compute_energies(q1, q2, q1_dot, q2_dot):
    # リンク1の運動エネルギー
    T1 = 0.5 * I1 * q1_dot**2 + 0.5 * m1 * lg1**2 * q1_dot**2

    # リンク2の運動エネルギー
    v2_x = l1 * np.cos(q1) * q1_dot + lg2 * np.cos(q1 + q2) * (q1_dot + q2_dot)
    v2_y = l1 * np.sin(q1) * q1_dot + lg2 * np.sin(q1 + q2) * (q1_dot + q2_dot)
    T2 = 0.5 * I2 * (q1_dot + q2_dot)**2 + 0.5 * m2 * (v2_x**2 + v2_y**2)

    # リンク1の位置エネルギー
    U1 = m1 * g * lg1 * np.sin(q1)

    # リンク2の位置エネルギー
    U2 = m2 * g * (l1 * np.sin(q1) + lg2 * np.sin(q1 + q2))

    # リンク1とリンク2の総エネルギー
    total_energy_link1 = T1 + U1
    total_energy_link2 = T2 + U2

    return total_energy_link1, total_energy_link2

def q_learning(runge_kutta):
    csv_file_path_max_R = os.path.join(save_dir1, f'max_R.csv')
    csv_file_path_episode_max_R = os.path.join(save_dir1, f'episode_max_R.csv')  # 新しく追加するCSVファイル
    max_rewards = []

    # エピソードごとの最大報酬を保存するCSVファイルの準備
    with open(csv_file_path_max_R, 'w', newline='') as csvfile_max_R, \
         open(csv_file_path_episode_max_R, 'w', newline='') as csvfile_episode_max_R:
        csv_writer2 = csv.writer(csvfile_max_R)
        csv_writer3 = csv.writer(csvfile_episode_max_R)  # エピソード内の最大報酬を保存するCSV
        csv_writer2.writerow(['episode', 'final_max_reward'])
        csv_writer3.writerow(['episode', 'max_reward_within_episode'])  # 新しいCSVファイルのヘッダー

        for episode in range(num_episodes):
            q1, q2, q1_dot, q2_dot = reset()
            q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = digitize_state(q1, q2, q1_dot, q2_dot)
            action = get_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, episode)

            max_reward = -float('inf')
            max_reward_step = 0
            max_reward_q_table = None
            cumulative_energy = 0  # 累積消費エネルギーを初期化

            # エピソードごとのデータを保存するCSVファイルの準備
            csv_file_path = os.path.join(save_dir2, f'try_{episode + 1}.csv')
            with open(csv_file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Time', 'Theta1', 'Theta2', 'Omega1', 'Omega2', 'v2', 'Theta_v', 'Reward', 'TotalEnergyLink1', 'TotalEnergyLink2', 'StepEnergy', 'CumulativeEnergy'])

                for i in range(max_number_of_steps):
                    q1, q2, q1_dot, q2_dot = runge_kutta(0, q1, q2, q1_dot, q2_dot, action, dt)
                    v_x2 = -l1 * np.sin(q1) * q1_dot - l2 * np.sin(q1 + q2) * (q1_dot + q2_dot)
                    v_y2 = l1 * np.cos(q1) * q1_dot + l2 * np.cos(q1 + q2) * (q1_dot + q2_dot)
                    v2 = np.sqrt(v_x2**2 + v_y2**2)

                    # 前腕リンクの角度が45度に近いかを評価
                    # q1_deg = np.degrees(q1)
                    # q2_deg = np.degrees(q2)

                    # θvを定義
                    # qv_deg = (q1_deg + q2_deg) - 90
                    qv = np.arctan2(v_x2, v_y2)

                    # θvを0〜360度の範囲に変換
                    # qv_deg = qv_deg % 360  # 360度で剰余をとる

                    # # 前腕リンクの角度が45度に近いかを評価
                    # qv = np.arctan2(v_x2, v_y2)

                    # # θvを0〜360度の範囲に変換
                    qv_deg = np.degrees(qv) % 360 

                    # トルクを取得
                    tau = np.zeros((2, 1))
                    if action == 0:
                        tau = np.array([[18.0], [0.0]])
                    elif action == 1:
                        tau = np.array([[-18.0], [0.0]])
                    elif action == 2:
                        tau = np.array([[0.0], [11.0]])
                    elif action == 3:
                        tau = np.array([[0.0], [-11.0]])
                    elif action == 4:
                        tau = np.array([[18.0], [11.0]])
                    elif action == 5:
                        tau = np.array([[-18.0], [-11.0]])
                    elif action == 6:
                        tau = np.array([[18.0], [-11.0]])
                    elif action == 7:
                        tau = np.array([[-18.0], [11.0]])
                    elif action == 8:
                        tau = np.array([[0.0], [0.0]])

                    # 各ステップでのリンク1とリンク2の消費エネルギーを計算
                    energy_consumed_link1 = np.abs(tau[0, 0] * q1_dot) * dt
                    energy_consumed_link2 = np.abs(tau[1, 0] * q2_dot) * dt
                    step_energy = energy_consumed_link1 + energy_consumed_link2  # ステップごとのエネルギー合計

                    # 累積エネルギーを更新
                    cumulative_energy += step_energy

                    # reward計算時に累積エネルギーをペナルティとして使用
                    reward = compute_reward(q1, q2, q1_dot, q2_dot, cumulative_energy)

                    # Qテーブルの更新
                    q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = digitize_state(q1, q2, q1_dot, q2_dot)
                    next_action = get_action(q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, episode)
                    next_q1, next_q2, next_q1_dot, next_q2_dot = runge_kutta(0, q1, q2, q1_dot, q2_dot, next_action, dt)

                    next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin = digitize_state(next_q1, next_q2, next_q1_dot, next_q2_dot)
                    Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action] += alpha * (reward + gamma * np.max(Q[next_q1_bin, next_q2_bin, next_q1_dot_bin, next_q2_dot_bin]) - Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action])

                    # データをCSVに記録
                    csv_writer.writerow([i * dt, q1 * 180 / np.pi, q2 * 180 / np.pi, q1_dot, q2_dot, v2, qv_deg % 360, reward, energy_consumed_link1, energy_consumed_link2, step_energy, cumulative_energy])

                    # 最大報酬の更新とQテーブルの保存
                    if reward > max_reward:
                        max_reward = reward
                        max_reward_step = i
                        max_reward_q_table = np.copy(Q)  # 現在のQテーブルを保存

                    q1 = next_q1
                    q2 = next_q2
                    q1_dot = next_q1_dot
                    q2_dot = next_q2_dot
                    action = next_action

                # エピソード内で達成された最大報酬を保存
                csv_writer3.writerow([episode + 1, max_reward])

                # 最大報酬が得られた時点のQテーブルを保存
                max_reward_q_table_path = os.path.join(save_dir3, f'max_reward_Q_table_episode_{episode + 1}_step_{max_reward_step + 1}.csv')
                with open(max_reward_q_table_path, 'w', newline='') as qfile:
                    q_writer = csv.writer(qfile)
                    q_writer.writerow(['q1_bin', 'q2_bin', 'q1_dot_bin', 'q2_dot_bin', 'action', 'value'])
                    for q1_bin in range(num_q1_bins):
                        for q2_bin in range(num_q2_bins):
                            for q1_dot_bin in range(num_q1_dot_bins):
                                for q2_dot_bin in range(num_q2_dot_bins):
                                    for action in range(num_actions):
                                        q_writer.writerow([q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action, max_reward_q_table[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action]])
                print(f'Max reward Q-table for episode {episode + 1} saved to {max_reward_q_table_path}')

                # エピソード終了時のQテーブルに基づいてfinal_max_rewardを再計算
                final_max_reward = -float('inf')
                q1, q2, q1_dot, q2_dot = reset()  # 状態をリセット

                for i in range(max_number_of_steps):
                    q1_bin, q2_bin, q1_dot_bin, q2_dot_bin = digitize_state(q1, q2, q1_dot, q2_dot)
                    
                    # エピソード終了時のQテーブルから最適なアクションを選択
                    optimal_action = np.argmax(Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin])
                    
                    q1, q2, q1_dot, q2_dot = runge_kutta(0, q1, q2, q1_dot, q2_dot, optimal_action, dt)
                    reward = compute_reward(q1, q2, q1_dot, q2_dot, cumulative_energy)
                    
                    # 最終的な最大報酬を更新
                    final_max_reward = max(final_max_reward, reward)

                # 保存した最大報酬時のQテーブルとは別に、エピソード終了時のQテーブルに基づいて再計算した報酬を保存
                max_rewards.append(final_max_reward)
                csv_writer2.writerow([episode + 1, final_max_reward])  # ここで保存
                print(f'Episode {episode + 1} final_max_reward: {final_max_reward}')

            # エピソード終了時のQテーブルを保存
            q_table_file_path = os.path.join(save_dir4, f'Q_table_episode_{episode + 1}.csv')
            with open(q_table_file_path, 'w', newline='') as qfile:
                q_writer = csv.writer(qfile)
                q_writer.writerow(['q1_bin', 'q2_bin', 'q1_dot_bin', 'q2_dot_bin', 'action', 'value'])
                for q1_bin in range(num_q1_bins):
                    for q2_bin in range(num_q2_bins):
                        for q1_dot_bin in range(num_q1_dot_bins):
                            for q2_dot_bin in range (num_q2_dot_bins):
                                for action in range(num_actions):
                                    q_writer.writerow([q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action, Q[q1_bin, q2_bin, q1_dot_bin, q2_dot_bin, action]])
            print(f'Q-table for episode {episode + 1} saved to {q_table_file_path}')

if __name__ == "__main__":
    # Q学習の実行
    q_learning(runge_kutta)