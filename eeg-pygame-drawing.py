import socket
import numpy as np
import pygame
import sys
import threading
import time
import json
import os
import random
from datetime import datetime

# 強化学習機能を持つEEGネットワーククラス
class RLEEGNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, weights=None):
        # 既存の重みで初期化、または新しい重みをランダムに生成
        if weights is not None:
            self.w1 = weights['w1']
            self.b1 = weights['b1']
            self.w2 = weights['w2']
            self.b2 = weights['b2']
        else:
            # 重みをランダムに初期化
            np.random.seed(42)  # 再現可能な結果のため
            self.w1 = np.random.randn(input_size, hidden_size) * 0.1
            self.b1 = np.zeros((1, hidden_size))
            self.w2 = np.random.randn(hidden_size, output_size) * 0.1
            self.b2 = np.zeros((1, output_size))
        
        # 強化学習パラメータ
        self.learning_rate = learning_rate
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        # 入力を行ベクトルに整形
        x = x.reshape(1, -1)
        
        # 第1層
        self.layer1_input = x
        self.layer1_z = np.dot(x, self.w1) + self.b1
        self.layer1_output = self.sigmoid(self.layer1_z)
        
        # 出力層
        self.layer2_z = np.dot(self.layer1_output, self.w2) + self.b2
        self.layer2_output = self.sigmoid(self.layer2_z)
        
        return self.layer2_output.flatten()  # 1D配列として返す
    
    def update_with_reward(self, target, reward_scale=0.1):
        """
        現在の出力と目標値の差に基づいて、リアルタイムで重みを更新する強化学習機能
        
        target: 期待される出力（例：[1, 0, 0, 1, 0] = オン、上、右）
        reward_scale: 報酬のスケーリング係数
        """
        # 出力誤差を計算
        target = np.array(target).reshape(1, -1)
        output_error = target - self.layer2_output
        
        # 報酬（正または負）に基づいて誤差をスケーリング
        reward = np.sum(output_error ** 2)  # 二乗誤差の合計
        error_scaled = output_error * reward_scale
        
        # 出力層の誤差勾配
        delta_output = error_scaled * self.sigmoid_derivative(self.layer2_output)
        
        # 隠れ層の誤差勾配
        delta_hidden = np.dot(delta_output, self.w2.T) * self.sigmoid_derivative(self.layer1_output)
        
        # 重みとバイアスの更新
        self.w2 += np.dot(self.layer1_output.T, delta_output) * self.learning_rate
        self.b2 += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate
        self.w1 += np.dot(self.layer1_input.T, delta_hidden) * self.learning_rate
        self.b1 += np.sum(delta_hidden, axis=0, keepdims=True) * self.learning_rate
        
        return reward
    
    def get_weights(self):
        return {
            'w1': self.w1.tolist(),
            'b1': self.b1.tolist(),
            'w2': self.w2.tolist(),
            'b2': self.b2.tolist()
        }
    
    def train_batch(self, training_data):
        """
        収集したデータサンプルを使用した簡単なバッチトレーニング方法
        """
        # 各状態の平均EEGパターンを計算
        state_averages = {}
        for state, samples in training_data.items():
            if samples:  # サンプルがある場合のみ処理
                # サンプルのリストをnumpy配列に変換
                samples_array = np.array(samples)
                # 平均を計算
                state_averages[state] = np.mean(samples_array, axis=0)
        
        # 必要な状態の平均が十分にある場合、それらを使用して重みを調整
        required_states = ['on', 'off', 'up', 'down', 'left', 'right']
        if all(state in state_averages for state in required_states):
            # これは重み調整の非常に簡略化されたアプローチ
            
            # 出力1（オン/オフ）の場合、オン状態とオフ状態の違いを認識するように調整
            on_off_diff = state_averages['on'] - state_averages['off']
            
            # 出力2-5（上、下、左、右）の場合、それぞれのパターンを認識するように調整
            direction_diffs = [
                state_averages['up'] - np.mean([state_averages[d] for d in ['down', 'left', 'right']], axis=0),
                state_averages['down'] - np.mean([state_averages[d] for d in ['up', 'left', 'right']], axis=0),
                state_averages['left'] - np.mean([state_averages[d] for d in ['up', 'down', 'right']], axis=0),
                state_averages['right'] - np.mean([state_averages[d] for d in ['up', 'down', 'left']], axis=0)
            ]
            
            # 非常に簡単な重み調整 - 実際には適切なトレーニングアルゴリズムが必要
            input_size = self.w1.shape[0]
            hidden_size = self.w1.shape[1]
            
            # 新しい重みを初期化 - ランダムな重みを完全に置き換える
            self.w1 = np.random.randn(input_size, hidden_size) * 0.01
            
            # 状態の違いを重みに組み込む
            for i in range(input_size):
                # オン/オフ検出用
                if i < len(on_off_diff):
                    self.w1[i, 0] += on_off_diff[i] * 0.1
                
                # 方向検出用
                for j, diff in enumerate(direction_diffs):
                    if i < len(diff):
                        self.w1[i, j+1] += diff[i] * 0.1
            
            # 出力層の重み - 簡略化されたアプローチ
            self.w2 = np.zeros_like(self.w2)
            for i in range(5):  # 5つの出力
                self.w2[i, i] = 1.0  # 直接マッピング
            
            return True  # トレーニング成功
        
        return False  # データ不足

# EEGデータをメッセージから解析
def parse_eeg_data(message):
    try:
        # 数値部分のみを抽出（「/EEG,ss」接頭辞を削除）
        if ",ss" in message:
            data_part = message.split(",ss")[1]
        else:
            data_part = message  # フォーマットが異なる場合
        
        # タイムスタンプが始まる場所を見つける（「2025-」で始まると仮定）
        timestamp_idx = data_part.find("2025-")
        if timestamp_idx != -1:
            data_part = data_part[:timestamp_idx]
        
        # セミコロンで分割して浮動小数点に変換
        # 変換前にnullバイトを削除
        values = [float(val.replace('\x00', '')) for val in data_part.split(";") if val.strip()]
        
        return np.array(values)
    except Exception as e:
        print(f"EEGデータの解析エラー: {e}")
        return None

# 神経ネットワークの出力を解釈
def interpret_outputs(outputs):
    if outputs is None or len(outputs) < 5:
        return {
            "status": "error",
            "message": "無効な出力"
        }
    
    # 出力1: バイナリオン/オフ（0.5閾値）
    on_off = "On" if outputs[0] >= 0.5 else "Off"
    
    # 出力2-5: 方向値（上、下、左、右）
    directions = ["up", "down", "left", "right"]
    direction_values = outputs[1:5]
    
    # 最も高い2つの方向値を取得
    sorted_indices = np.argsort(direction_values)[::-1]
    primary_direction = directions[sorted_indices[0]]
    secondary_direction = directions[sorted_indices[1]]
    primary_value = direction_values[sorted_indices[0]]
    secondary_value = direction_values[sorted_indices[1]]
    
    return {
        "status": "success",
        "on_off": on_off,
        "primary_direction": primary_direction,
        "primary_value": primary_value,
        "secondary_direction": secondary_direction,
        "secondary_value": secondary_value,
        "raw_outputs": outputs,
        "timestamp": datetime.now()
    }

# 強化学習用のランダムタスク生成関数
def generate_random_task():
    """
    ランダムなトレーニングタスクを生成する
    タスクは[on/off, up, down, left, right]の組み合わせ
    戻り値: (タスク説明, 目標出力配列)
    """
    # オン/オフ状態をランダムに決定
    on_off = random.choice(["On", "Off"])
    
    # 主要方向をランダムに決定
    primary_dir = random.choice(["up", "down", "left", "right"])
    
    # 目標出力配列を作成 [on/off, up, down, left, right]
    target = [1 if on_off == "On" else 0, 0, 0, 0, 0]
    
    # 方向インデックスをセット
    dir_indices = {"up": 1, "down": 2, "left": 3, "right": 4}
    target[dir_indices[primary_dir]] = 1
    
    # タスク説明を作成
    description = f"{primary_dir.upper()} および {on_off}"
    
    return description, target

# グローバル変数
latest_outputs = None
output_timestamp = None
connection_status = "disconnected"
drawing_history = []
current_history_index = -1
latest_eeg_data = None
training_data = {
    'on': [],
    'off': [],
    'up': [],
    'down': [],
    'left': [],
    'right': []
}
current_training_state = None
nn = None  # 神経ネットワークインスタンス

# 強化学習用変数
rl_task_description = ""
rl_target_output = None
rl_task_start_time = 0
rl_reward_history = []
rl_is_training = False

# トレーニングデータをファイルに保存する関数
def save_training_data(filename="brain_canvas_training.json"):
    global training_data
    try:
        # JSON シリアル化のために numpy 配列をリストに変換
        serializable_data = {}
        for state, samples in training_data.items():
            serializable_data[state] = [sample.tolist() for sample in samples]
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f)
        
        print(f"トレーニングデータを {filename} に保存しました")
        return True
    except Exception as e:
        print(f"トレーニングデータの保存エラー: {e}")
        return False

# ファイルからトレーニングデータを読み込む関数
def load_training_data(filename="brain_canvas_training.json"):
    global training_data
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                loaded_data = json.load(f)
            
            # リストをnumpy配列に戻す
            for state, samples in loaded_data.items():
                training_data[state] = [np.array(sample) for sample in samples]
            
            print(f"{filename} からトレーニングデータを読み込みました")
            return True
        else:
            print(f"トレーニングファイル {filename} が見つかりません")
            return False
    except Exception as e:
        print(f"トレーニングデータの読み込みエラー: {e}")
        return False

# ニューラルネットワークの重みを保存する関数
def save_network_weights(nn, filename="brain_canvas_weights.json"):
    try:
        weights = nn.get_weights()
        with open(filename, 'w') as f:
            json.dump(weights, f)
        
        print(f"ニューラルネットワークの重みを {filename} に保存しました")
        return True
    except Exception as e:
        print(f"ニューラルネットワークの重み保存エラー: {e}")
        return False

# ニューラルネットワークの重みを読み込む関数
def load_network_weights(input_size, hidden_size, output_size, filename="brain_canvas_weights.json"):
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                weights = json.load(f)
            
            # リストをnumpy配列に戻す
            for key in weights:
                weights[key] = np.array(weights[key])
            
            nn = RLEEGNetwork(input_size, hidden_size, output_size, weights=weights)
            print(f"{filename} からニューラルネットワークの重みを読み込みました")
            return nn
        else:
            print(f"重みファイル {filename} が見つかりません、ランダムな重みで初期化します")
            return RLEEGNetwork(input_size, hidden_size, output_size)
    except Exception as e:
        print(f"ニューラルネットワークの重み読み込みエラー: {e}")
        return RLEEGNetwork(input_size, hidden_size, output_size)

# UDPを介してEEGデータを受信する関数
def udp_receiver(ip_address, port_number):
    global latest_outputs, output_timestamp, connection_status, latest_eeg_data, nn
    global rl_is_training, rl_task_description, rl_target_output, rl_task_start_time, rl_reward_history
    
    # UDPソケットを作成
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # ニューラルネットワーク設定を初期化
    hidden_size = 10
    output_size = 5  # 1はオン/オフ用、4は方向用（上、下、左、右）
    
    try:
        s.bind((ip_address, port_number))
        print(f"UDP {ip_address}:{port_number} でEEGデータを待機中")
        connection_status = "waiting"
        
        while True:
            # データを受信（最大1024バイト）
            data, addr = s.recvfrom(1024)
            
            # メッセージをデコードして解析
            message = data.decode()
            print(f"\n{addr[0]}:{addr[1]} からメッセージを受信")
            print(f"生のメッセージ: {message}")
            print(f"メッセージサイズ: {len(data)} バイト")
            
            # EEGデータを解析
            eeg_data = parse_eeg_data(message)
            
            if eeg_data is not None and len(eeg_data) > 0:
                # 最新のEEGデータを保存（トレーニング用）
                latest_eeg_data = eeg_data
                
                # ネットワークが初めてのデータの場合、または入力サイズが変わった場合に初期化
                if nn is None:
                    input_size = len(eeg_data)
                    print(f"入力サイズ: {input_size} でニューラルネットワークを初期化")
                    
                    # 保存された重みを最初に読み込む
                    nn = load_network_weights(input_size, hidden_size, output_size)
                elif nn.w1.shape[0] != len(eeg_data):
                    # 入力サイズが変わった場合、ネットワークを再初期化
                    input_size = len(eeg_data)
                    print(f"新しい入力サイズ: {input_size} でニューラルネットワークを再初期化")
                    nn = RLEEGNetwork(input_size, hidden_size, output_size)
                
                # データを処理
                output = nn.forward(eeg_data)
                latest_outputs = output
                output_timestamp = datetime.now()
                connection_status = "connected"
                
                # 出力と解釈を表示
                interpretation = interpret_outputs(output)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] ニューラルネットワーク出力:")
                print(f"  オン/オフ: {interpretation['on_off']}")
                print(f"  主方向: {interpretation['primary_direction']} ({interpretation['primary_value']:.4f})")
                print(f"  副方向: {interpretation['secondary_direction']} ({interpretation['secondary_value']:.4f})")
                
                # 強化学習トレーニング中の場合、リアルタイムでモデルを更新
                if rl_is_training and rl_target_output is not None:
                    current_time = time.time()
                    # タスク開始から10秒経過したら新しいタスクを生成
                    if current_time - rl_task_start_time >= 10:
                        rl_task_description, rl_target_output = generate_random_task()
                        rl_task_start_time = current_time
                        print(f"新しいタスク: {rl_task_description}")
                    
                    # モデルを更新し、報酬を記録
                    reward = nn.update_with_reward(rl_target_output)
                    rl_reward_history.append(reward)
                    print(f"報酬: {reward:.6f}")
            
    except KeyboardInterrupt:
        print("\nUDPサーバー停止中...")
    except Exception as e:
        print(f"エラー: {e}")
        connection_status = "error"
    finally:
        # ソケットを閉じる
        s.close()
        print("ソケット閉じました")

# pygameを初期化
pygame.init()

# 定数
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
CANVAS_WIDTH = 900
CANVAS_HEIGHT = 600
SIDEBAR_WIDTH = 300
FONT_SIZE = 20
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 40
BUTTON_MARGIN = 10

# 色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)

# メイン関数
def main():
    global latest_outputs, output_timestamp, connection_status, drawing_history, current_history_index
    global training_data, current_training_state, latest_eeg_data, nn
    global rl_is_training, rl_task_description, rl_target_output, rl_task_start_time, rl_reward_history
    
    # 画面を作成
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BrainCanvas - EEG描画アプリ（強化学習対応）")
    
    # フォントを作成
    font = pygame.font.Font("/Users/daist/NotoSansJP-VariableFont_wght.ttf", FONT_SIZE)
    large_font = pygame.font.SysFont(None, FONT_SIZE * 2)
    
    # キャンバスサーフェスを作成
    canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
    canvas.fill(WHITE)
    
    # 履歴に初期キャンバス状態を追加
    drawing_history.append(pygame.Surface.copy(canvas))
    current_history_index = 0
    
    # UPDレシーバースレッドを開始
    ip_address = '192.168.0.247'  # デフォルトIP
    port_number = 8001  # デフォルトポート
    
    udp_thread = threading.Thread(target=udp_receiver, args=(ip_address, port_number))
    udp_thread.daemon = True
    udp_thread.start()
    
    # 描画状態変数
    drawing_mode = "manual"  # "manual", "auto", "training", "rl_training"
    tool = "pencil"  # or "eraser"
    cursor_pos = [CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2]
    is_drawing = False
    
    # トレーニング変数
    training_sequence = ['on', 'off', 'up', 'down', 'left', 'right']
    training_index = 0
    training_samples_per_state = 5
    training_current_samples = 0
    training_timer = 0
    training_state = "ready"  # "ready", "collect", "complete"
    
    # ボタン
    buttons = {
        "pencil": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT),
        "eraser": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN + BUTTON_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT),
        "undo": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*2 + BUTTON_HEIGHT, BUTTON_WIDTH, BUTTON_HEIGHT),
        "clear": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN + BUTTON_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*2 + BUTTON_HEIGHT, BUTTON_WIDTH, BUTTON_HEIGHT),
        "mode": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*3 + BUTTON_HEIGHT*2, BUTTON_WIDTH*2 + BUTTON_MARGIN, BUTTON_HEIGHT),
        "save": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*4 + BUTTON_HEIGHT*3, BUTTON_WIDTH, BUTTON_HEIGHT),
        "train_start": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*5 + BUTTON_HEIGHT*4, BUTTON_WIDTH*2 + BUTTON_MARGIN, BUTTON_HEIGHT),
        "save_training": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*6 + BUTTON_HEIGHT*5, BUTTON_WIDTH, BUTTON_HEIGHT),
        "load_training": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN + BUTTON_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*6 + BUTTON_HEIGHT*5, BUTTON_WIDTH, BUTTON_HEIGHT),
        "rl_start": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*7 + BUTTON_HEIGHT*6, BUTTON_WIDTH*2 + BUTTON_MARGIN, BUTTON_HEIGHT),
    }
    
    # メインループ
    running = True
    clock = pygame.time.Clock()
    last_auto_save = time.time()
    
    while running:
        # イベント処理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左クリック
                    # マウスがキャンバス上にあるか確認
                    if event.pos[0] < CANVAS_WIDTH and drawing_mode == "manual":
                        is_drawing = True
                        cursor_pos = list(event.pos)
                    
                    # マウスがボタン上にあるか確認
                    if buttons["pencil"].collidepoint(event.pos):
                        tool = "pencil"
                    elif buttons["eraser"].collidepoint(event.pos):
                        tool = "eraser"
                    elif buttons["undo"].collidepoint(event.pos):
                        if current_history_index > 0:
                            current_history_index -= 1
                            canvas = pygame.Surface.copy(drawing_history[current_history_index])
                    elif buttons["clear"].collidepoint(event.pos):
                        canvas.fill(WHITE)
                        drawing_history = [pygame.Surface.copy(canvas)]
                        current_history_index = 0
                    elif buttons["mode"].collidepoint(event.pos):
                        # モード切り替え: manual -> auto -> training -> rl_training -> manual
                        if drawing_mode == "manual":
                            drawing_mode = "auto"
                        elif drawing_mode == "auto":
                            drawing_mode = "training"
                            training_state = "ready"
                            training_index = 0
                            training_current_samples = 0
                        elif drawing_mode == "training":
                            drawing_mode = "rl_training"
                            rl_is_training = False
                        else:
                            drawing_mode = "manual"
                    elif buttons["save"].collidepoint(event.pos):
                        save_path = f"eeg_drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        pygame.image.save(canvas, save_path)
                        print(f"キャンバスを {save_path} に保存しました")
                    elif buttons["train_start"].collidepoint(event.pos) and drawing_mode == "training":
                        if training_state == "ready":
                            training_state = "collect"
                            training_timer = time.time()
                        elif training_state == "complete":
                            # ニューラルネットワークをトレーニング
                            if nn is not None and nn.train_batch(training_data):
                                # トレーニングされた重みを保存
                                save_network_weights(nn)
                                training_state = "ready"
                                training_index = 0
                                training_current_samples = 0
                    elif buttons["save_training"].collidepoint(event.pos):
                        save_training_data()
                    elif buttons["load_training"].collidepoint(event.pos):
                        if load_training_data():
                            # 読み込んだデータでネットワークを再トレーニング
                            if nn is not None:
                                nn.train_batch(training_data)
                                save_network_weights(nn)
                    elif buttons["rl_start"].collidepoint(event.pos) and drawing_mode == "rl_training":
                        # 強化学習のオン/オフを切り替え
                        rl_is_training = not rl_is_training
                        if rl_is_training:
                            # 新しいランダムタスクを開始
                            rl_task_description, rl_target_output = generate_random_task()
                            rl_task_start_time = time.time()
                            rl_reward_history = []
                        else:
                            # トレーニング停止時に重みを保存
                            if nn is not None:
                                save_network_weights(nn)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and is_drawing:  # 左クリック解除
                    is_drawing = False
                    # 履歴に保存
                    if current_history_index < len(drawing_history) - 1:
                        drawing_history = drawing_history[:current_history_index+1]
                    drawing_history.append(pygame.Surface.copy(canvas))
                    current_history_index = len(drawing_history) - 1
            
            elif event.type == pygame.MOUSEMOTION:
                if is_drawing and drawing_mode == "manual":
                    # 現在位置を取得
                    x, y = event.pos
                    if x >= CANVAS_WIDTH:
                        x = CANVAS_WIDTH - 1
                    
                    # 前の位置から現在位置まで線を描画
                    color = BLACK if tool == "pencil" else WHITE
                    line_width = 2 if tool == "pencil" else 20
                    pygame.draw.line(canvas, color, cursor_pos, (x, y), line_width)
                    cursor_pos = [x, y]
        
        # 自動描画モードの処理
        if drawing_mode == "auto" and latest_outputs is not None:
            interpretation = interpret_outputs(latest_outputs)
            
            if interpretation["status"] == "success":
                # EEG出力に基づいてカーソルを移動
                move_step = 1  # フレームあたりのピクセル数
                
                # 主方向
                if interpretation["primary_direction"] == "up":
                    cursor_pos[1] -= move_step * interpretation["primary_value"]
                elif interpretation["primary_direction"] == "down":
                    cursor_pos[1] += move_step * interpretation["primary_value"]
                elif interpretation["primary_direction"] == "left":
                    cursor_pos[0] -= move_step * interpretation["primary_value"]
                elif interpretation["primary_direction"] == "right":
                    cursor_pos[0] += move_step * interpretation["primary_value"]
                
                # 副方向
                if interpretation["secondary_direction"] == "up":
                    cursor_pos[1] -= move_step * interpretation["secondary_value"]
                elif interpretation["secondary_direction"] == "down":
                    cursor_pos[1] += move_step * interpretation["secondary_value"]
                elif interpretation["secondary_direction"] == "left":
                    cursor_pos[0] -= move_step * interpretation["secondary_value"]
                elif interpretation["secondary_direction"] == "right":
                    cursor_pos[0] += move_step * interpretation["secondary_value"]
                
                # カーソルがキャンバス境界内に収まるようにする
                cursor_pos[0] = max(0, min(CANVAS_WIDTH-1, cursor_pos[0]))
                cursor_pos[1] = max(0, min(CANVAS_HEIGHT-1, cursor_pos[1]))
                
                # "On"なら描画
                if interpretation["on_off"] == "On":
                    color = BLACK if tool == "pencil" else WHITE
                    line_width = 2 if tool == "pencil" else 20
                    
                    # カーソル位置にドットを描画
                    pygame.draw.circle(canvas, color, (int(cursor_pos[0]), int(cursor_pos[1])), line_width // 2)
                
                # 自動モードで定期的に保存
                current_time = time.time()
                if current_time - last_auto_save > 5:  # 自動モードで5秒ごとに保存
                    if current_history_index < len(drawing_history) - 1:
                        drawing_history = drawing_history[:current_history_index+1]
                    drawing_history.append(pygame.Surface.copy(canvas))
                    current_history_index = len(drawing_history) - 1
                    last_auto_save = current_time
        
        # トレーニングモードの処理
        if drawing_mode == "training" and training_state == "collect" and latest_eeg_data is not None:
            current_state = training_sequence[training_index]
            current_training_state = current_state
            
            # データサンプルを収集
            current_time = time.time()
            # サンプル間に3秒の安定時間を設ける
            if current_time - training_timer > 3:
                # 現在のEEGデータをトレーニングセットに追加
                training_data[current_state].append(latest_eeg_data)
                training_current_samples += 1
                training_timer = current_time
                
                # 現在の状態に対して十分なサンプルがあるか確認
                if training_current_samples >= training_samples_per_state:
                    training_index += 1
                    training_current_samples = 0
                    
                    # トレーニングが完了したか確認
                    if training_index >= len(training_sequence):
                        training_state = "complete"
                        current_training_state = None
        
        # 画面を描画
        screen.fill(LIGHT_GRAY)
        
        # キャンバスの背景ボーダーを描画
        pygame.draw.rect(screen, BLACK, (0, 0, CANVAS_WIDTH, CANVAS_HEIGHT), 1)
        
        # キャンバスを描画
        screen.blit(canvas, (0, 0))
        
        # 自動モードでカーソルを描画
        if drawing_mode == "auto":
            cursor_color = RED if latest_outputs is not None and latest_outputs[0] >= 0.5 else YELLOW
            pygame.draw.circle(screen, cursor_color, (int(cursor_pos[0]), int(cursor_pos[1])), 5, 1)
        
        # トレーニングモードでトレーニング指示を描画
        if drawing_mode == "training" and training_state == "collect":
            # 現在の指示でオーバーレイを描画
            overlay = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))  # 半透明の黒
            screen.blit(overlay, (0, 0))
            
            current_state = training_sequence[training_index]
            
            # 指示を表示
            if current_state == "on":
                instruction = "描画について考えてください（オン状態）"
            elif current_state == "off":
                instruction = "描画しないことについて考えてください（オフ状態）"
            else:
                instruction = f"{current_state.upper()} 方向に移動することを考えてください"
            
            text = large_font.render(instruction, True, WHITE)
            text_rect = text.get_rect(center=(CANVAS_WIDTH//2, CANVAS_HEIGHT//2))
            screen.blit(text, text_rect)
            
            # サンプルカウンターを表示
            counter_text = font.render(f"{current_state} のサンプル {training_current_samples + 1}/{training_samples_per_state}", True, WHITE)
            counter_rect = counter_text.get_rect(center=(CANVAS_WIDTH//2, CANVAS_HEIGHT//2 + 40))
            screen.blit(counter_text, counter_rect)
            
            # カウントダウンを表示
            countdown = max(0, int(3 - (time.time() - training_timer)))
            countdown_text = large_font.render(str(countdown), True, WHITE)
            countdown_rect = countdown_text.get_rect(center=(CANVAS_WIDTH//2, CANVAS_HEIGHT//2 + 80))
            screen.blit(countdown_text, countdown_rect)
        
        # 強化学習トレーニングモードの指示を描画
        if drawing_mode == "rl_training" and rl_is_training:
            # 現在のタスク指示でオーバーレイを描画
            overlay = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))  # 半透明の黒
            screen.blit(overlay, (0, 0))
            
            # タスク指示を表示
            text = large_font.render(f"タスク: {rl_task_description}", True, WHITE)
            text_rect = text.get_rect(center=(CANVAS_WIDTH//2, CANVAS_HEIGHT//2))
            screen.blit(text, text_rect)
            
            # 残り時間を表示
            remaining = max(0, int(10 - (time.time() - rl_task_start_time)))
            countdown_text = large_font.render(f"残り時間: {remaining}秒", True, WHITE)
            countdown_rect = countdown_text.get_rect(center=(CANVAS_WIDTH//2, CANVAS_HEIGHT//2 + 40))
            screen.blit(countdown_text, countdown_rect)
            
            # 現在の報酬を表示（あれば）
            if rl_reward_history:
                reward_text = font.render(f"現在の報酬: {rl_reward_history[-1]:.6f}", True, WHITE)
                reward_rect = reward_text.get_rect(center=(CANVAS_WIDTH//2, CANVAS_HEIGHT//2 + 80))
                screen.blit(reward_text, reward_rect)
        
        # サイドバーを描画
        pygame.draw.rect(screen, GRAY, (CANVAS_WIDTH, 0, SIDEBAR_WIDTH, SCREEN_HEIGHT))
        
        # ボタンを描画
        for btn_name, btn_rect in buttons.items():
            # トレーニング固有のボタンがトレーニングモードでない場合はスキップ
            if btn_name.startswith("train_") and drawing_mode != "training":
                continue
            # 強化学習固有のボタンが強化学習モードでない場合はスキップ
            if btn_name.startswith("rl_") and drawing_mode != "rl_training":
                continue
            
            if btn_name == "mode":
                if drawing_mode == "auto":
                    color = BLUE
                elif drawing_mode == "training":
                    color = PURPLE
                elif drawing_mode == "rl_training":
                    color = ORANGE
                else:
                    color = GREEN
            elif btn_name == "train_start":
                if training_state == "ready":
                    color = GREEN
                elif training_state == "collect":
                    color = YELLOW
                elif training_state == "complete":
                    color = PURPLE
                else:
                    color = GRAY
            elif btn_name == "rl_start":
                color = GREEN if not rl_is_training else RED
            elif btn_name == tool:
                color = BLUE
            else:
                color = GRAY
            
            pygame.draw.rect(screen, color, btn_rect)
            
            # ボタンテキスト
            if btn_name == "mode":
                label = f"モード: {drawing_mode.upper()}"
            elif btn_name == "train_start":
                if training_state == "ready":
                    label = "トレーニング開始"
                elif training_state == "collect":
                    label = "トレーニング中..."
                elif training_state == "complete":
                    label = "モデル保存"
                else:
                    label = "トレーニング"
            elif btn_name == "rl_start":
                label = "RL停止" if rl_is_training else "RL開始"
            else:
                label = btn_name.replace("_", " ").capitalize()
            
            text = font.render(label, True, WHITE)
            text_rect = text.get_rect(center=btn_rect.center)
            screen.blit(text, text_rect)
        
        # 接続状態を描画
        status_text = f"状態: {connection_status}"
        status_color = GREEN if connection_status == "connected" else YELLOW if connection_status == "waiting" else RED
        status_surf = font.render(status_text, True, status_color)
        screen.blit(status_surf, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 150))
        
        # NN出力の可視化
        if latest_outputs is not None:
            interpretation = interpret_outputs(latest_outputs)
            
            if interpretation["status"] == "success":
                # "オン/オフ"状態を描画
                on_off_text = f"状態: {interpretation['on_off']}"
                on_off_color = GREEN if interpretation['on_off'] == 'On' else GRAY
                on_off_surf = font.render(on_off_text, True, on_off_color)
                screen.blit(on_off_surf, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 120))
                
                # 主方向を描画
                primary_text = f"主方向: {interpretation['primary_direction']} ({interpretation['primary_value']:.2f})"
                primary_surf = font.render(primary_text, True, BLACK)
                screen.blit(primary_surf, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 90))
                
                # 主方向バーを描画
                bar_width = int(interpretation['primary_value'] * 200)
                pygame.draw.rect(screen, BLUE, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 70, bar_width, 10))
                pygame.draw.rect(screen, BLACK, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 70, 200, 10), 1)
                
                # 副方向を描画
                secondary_text = f"副方向: {interpretation['secondary_direction']} ({interpretation['secondary_value']:.2f})"
                secondary_surf = font.render(secondary_text, True, BLACK)
                screen.blit(secondary_surf, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 50))
                
                # 副方向バーを描画
                bar_width = int(interpretation['secondary_value'] * 200)
                pygame.draw.rect(screen, GREEN, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 30, bar_width, 10))
                pygame.draw.rect(screen, BLACK, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 30, 200, 10), 1)
                
                # 生の値を描画
                for i, value in enumerate(interpretation['raw_outputs']):
                    bar_height = int(value * 60)
                    bar_rect = pygame.Rect(
                        CANVAS_WIDTH + 10 + i * 40, 
                        CANVAS_HEIGHT - 200 - bar_height, 
                        30, 
                        bar_height
                    )
                    pygame.draw.rect(screen, (100, 100, 255), bar_rect)
                    pygame.draw.rect(screen, BLACK, bar_rect, 1)
                    
                    # ラベル
                    output_label = font.render(str(i), True, BLACK)
                    screen.blit(output_label, (CANVAS_WIDTH + 20 + i * 40, CANVAS_HEIGHT - 190))
        
        # トレーニングモードでトレーニングデータの統計を描画
        if drawing_mode == "training":
            y_pos = 250
            training_stats_title = font.render("トレーニングデータ統計:", True, BLACK)
            screen.blit(training_stats_title, (CANVAS_WIDTH + 10, y_pos))
            y_pos += 25
            
            for state, samples in training_data.items():
                samples_text = font.render(f"{state}: {len(samples)} サンプル", True, BLACK)
                screen.blit(samples_text, (CANVAS_WIDTH + 10, y_pos))
                y_pos += 20
        
        # 強化学習モードで報酬履歴を描画
        if drawing_mode == "rl_training":
            y_pos = 250
            rl_stats_title = font.render("強化学習状態:", True, BLACK)
            screen.blit(rl_stats_title, (CANVAS_WIDTH + 10, y_pos))
            y_pos += 25
            
            rl_status = font.render(f"状態: {'トレーニング中' if rl_is_training else '停止'}", True, BLACK)
            screen.blit(rl_status, (CANVAS_WIDTH + 10, y_pos))
            y_pos += 20
            
            if rl_task_description:
                rl_task = font.render(f"タスク: {rl_task_description}", True, BLACK)
                screen.blit(rl_task, (CANVAS_WIDTH + 10, y_pos))
                y_pos += 20
            
            if rl_reward_history:
                rl_reward = font.render(f"最後の報酬: {rl_reward_history[-1]:.6f}", True, BLACK)
                screen.blit(rl_reward, (CANVAS_WIDTH + 10, y_pos))
                y_pos += 20
                
                avg_reward = sum(rl_reward_history[-10:]) / min(10, len(rl_reward_history))
                rl_avg = font.render(f"平均報酬(10): {avg_reward:.6f}", True, BLACK)
                screen.blit(rl_avg, (CANVAS_WIDTH + 10, y_pos))
        
        # ディスプレイを更新
        pygame.display.flip()
        
        # フレームレートを制限
        clock.tick(60)
    
    # トレーニングが行われた場合、終了前にトレーニングデータを保存
    if any(len(samples) > 0 for samples in training_data.values()):
        save_training_data()
    
    # pygameを終了
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()