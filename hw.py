#!/usr/bin/python3
#*-* coding: utf-8 -*-
# nohup python -u rl.py > log.log &

# 라이브러리
import gym
import collections
import random
import pandas as pd
import numpy as np

# Plot 라이브러리
import matplotlib.pyplot as plt

# GYM 버전 확인
from packaging.specifiers import SpecifierSet
GYM_COMPARE = SpecifierSet(">=0.24,<0.26")
GYM_OLD = gym.__version__ in GYM_COMPARE

# GYM 동영상 저장 Library
import imageio

# PyTorch 라이브러리
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 런타임 환경관련 라이브러리
import os
import warnings
import signal
import sys

# Time 라이브러리
from time import time

# DeprecationWarning 메세지 제외
warnings.filterwarnings("ignore", category=DeprecationWarning)
# KeyboardInterrupt 발생 시 Traceback 제외
signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

# GYM 환경설정
TEST = True            # Test(테스트 모드) 여부, True:Global에서 지정한 Hyper parameter만 수행, False:테스트 대상 Hyper parameter 전체에 대해 수행

REPLAY = True          # Replay(모델 테스트) 여부, True:저장된 모델 테스트, False:모델 학습
RECORD = True          # 모델을 영상으로 저장 여부, True:영상 저장(첫 5개 Episode), False:영상 저장하지 않음
GYM = 'LunarLander-v2' # OpenAI GYM 명칭
WIND = True            # WIND(Stochastic environment) 사용여부
SEED = 10              # SEED
TARGET_RETURN = 200    # 목표값, 그래프용
STRETCH_GOAL = 250     # 도전목표값, 그래프용

# Hyper parameters
FC1_SIZE = 128         # Hidden layer 1 크기
FC2_SIZE = 128         # Hidden layer 2 크기
BUFFER_SIZE = 50000    # ReplayBuffer 크기
BATCH_SIZE = 64        # Minibatch 크기
DROPOUT = 0.0          # Dropout
GAMMA = 0.99           # 할인율 (discount factor)
TAU = 0.01             # 학습 모델 Soft update(q → q_target) 비율
LEARNING_RATE = 0.0005 # 학습율
EPS_START = 0.5        # Epsilon 시작 비율
EPS_END = 0.01         # Epsilon 최종 비율
EPS_DECAY = 0.995      # Epsilon 감소율

UPDATE_EVERY = 4       # 학습 주기
UPDATE_RATIO = 1.0     # Replay Buffer에서 업데이트를 시작할 비율 (BATCH_SIZE 기준)
TOTAL_EPISODES = 2000  # 최대 Episode 개수
MAX_TIMESTAMP = 1000   # Episode당 최대 Timestamp

# 알고리즘
USE_LINEAR_EPS_DECAY = False
USE_DOUBLE_DQN = True
USE_DUELING_DQN = True

# Model import/export
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(SCRIPT_PATH, 'model')
SCORE_PATH = os.path.join(SCRIPT_PATH, 'score')
MP4_PATH = os.path.join(SCRIPT_PATH, 'mp4')
os.makedirs(MODEL_PATH) if not os.path.exists(MODEL_PATH) else None
os.makedirs(SCORE_PATH) if not os.path.exists(SCORE_PATH) else None
os.makedirs(MP4_PATH) if not os.path.exists(MP4_PATH) else None




# Q Deep Network 클래스
class QNetwork(nn.Module):
    # Class 초기화
    def __init__(self, state_size, action_size, dropout=0.6, seed=0, fc1_size=64, fc2_size=64):
        # nn의 기본 __init__ 수행
        super(QNetwork, self).__init__()
        # 관련된 변수 초기화
        self.state_size = state_size
        self.action_size = action_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        # 동일한 결과 재현을 위한 SEED 값 설정
        self.seed = torch.manual_seed(seed)
        # FC1 설정 (Input Layer(observation space) → Hidden layer 1)
        self.fc1 = nn.Linear(state_size, fc1_size)
        # FC1 설정 (Hidden layer 1 → Hidden layer 2)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        # FC1 설정 (Hidden layer 2 → Output layer(action space)))
        self.fc3 = nn.Linear(fc2_size, action_size)
        # Dropout 설정
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    # Forward
    def forward(self, x):
        # FC1 수행, activation function : ReLU
        x = self.fc1(x)
        if self.dropout:
            x = self.dropout1(x)
        x = F.relu(x)
        # FC2 수행, activation function : ReLU
        x = self.fc2(x)
        if self.dropout:
            x = self.dropout2(x)
        x = F.relu(x)
        # FC3 수행
        x = self.fc3(x)

        # output 결과 리턴
        return x



# Dueling Q Deep Network 클래스
class DuelingQNetwork(nn.Module):
    # Class 초기화
    def __init__(self, state_size, action_size, dropout=0.6, seed=0, fc1_size=64, fc2_size=64):
        # nn의 기본 __init__ 수행
        super(DuelingQNetwork, self).__init__()
        # 관련된 변수 초기화
        self.state_size = state_size
        self.action_size = action_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        # 동일한 결과 재현을 위한 SEED 값 설정
        self.seed = torch.manual_seed(seed)
        # FC1 설정 (Input Layer(observation space) → Hidden layer 1)
        self.fc1 = nn.Linear(state_size, fc1_size)
        #####
        self.fc_value = nn.Linear(fc1_size, fc2_size)
        self.fc_adv = nn.Linear(fc1_size, fc2_size)
        self.value = nn.Linear(fc2_size, 1)
        self.adv = nn.Linear(fc2_size, action_size)
        # Dropout 설정
        self.dropout = dropout
        self.dropoutv = nn.Dropout(p=dropout)
        self.dropouta = nn.Dropout(p=dropout)
        self.dropoutx = nn.Dropout(p=dropout)

    # Forward
    def forward(self, x):
        x = self.fc1(x)
        if self.dropout:
            x = self.dropoutx(x)
        x = F.relu(x)
        v = self.fc_value(x)
        a = self.fc_adv(x)
        if self.dropout:
            v = self.dropoutv(x)
            a = self.dropouta(x)
        v = F.relu(v)
        a = F.relu(a)
        v = self.value(v)
        a = self.adv(a)
        a_avg = torch.mean(a)
        q = v + a - a_avg

        return q



# Replay Buffer 클래스
class ReplayBuffer():
    # Class 초기화
    def __init__(self, action_size, buffer_size, batch_size, seed=0):
        # Action size
        self.action_size = action_size
        # Batch size
        self.batch_size = batch_size
        # 저장용 Double-ended 큐 생성 (사이즈만 정하면 나머지는 신경쓸 필요가 없음)
        self.memory = collections.deque(maxlen=buffer_size)
        # 저장을 위한 구조 생성 (이름 기준으로 사용하기 위해 Named Tuple 구조 사용)
        self.experience = collections.namedtuple("exp", field_names=["state", "action", "reward", "next_state", "done"])
        # 동일한 결과 재현을 위한 SEED 값 설정
        self.seed = random.seed(seed)

    # ReplayBuffer에 추가
    def add(self, state, action, reward, next_state, done):
        # Named tuple 형태로 값 할당
        e = self.experience(state, action, reward, next_state, done)
        # deque에 추가
        self.memory.append(e)

    # Sampling 함수
    def sample(self):
        # deque에서 batch_size 만큼 랜덤하게 가져옴
        experiences = random.sample(self.memory, self.batch_size)

        # experiences array에서 원하는 값들을 추출하여 np.array로 만들고, 이것을 torch 형태로 변환함
        # torch.Tensor(x) 는 데이터 형변환이 생겨서 향후 처리과정에서 dtype 이슈가 발생할 가능성이 높음
        # torch.from_numpy(x) 로 수행함
        # 최종 값으로 변환하여 문제 발생 가능성 제거
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float()

        # 가져온 값 리턴
        return states, actions, rewards, next_states, dones

    # Size (len(memory) 로 호출가능)
    def __len__(self):
        return len(self.memory)



# Agent 클래스
class Agent():
    # Class 초기화
    def __init__(self, state_size, action_size, configure):
        # 나머지 수행에 필요한 값 초기화
        self.state_size = state_size
        self.action_size = action_size
        self.gym_old = configure.gym_old
        self.replay = configure.replay
        self.record = configure.record
        self.gym = configure.gym
        self.wind = configure.wind
        self.seed = configure.seed
        self.target_return = configure.target_return
        self.stretch_goal = configure.stretch_goal
        self.fc1_size = configure.fc1_size
        self.fc2_size = configure.fc2_size
        self.buffer_size = configure.buffer_size
        self.batch_size = configure.batch_size
        self.dropout = configure.dropout
        self.gamma = configure.gamma
        self.tau = configure.tau
        self.learning_rate = configure.learning_rate
        self.eps_start = configure.eps_start
        self.eps_end = configure.eps_end
        self.eps_decay = configure.eps_decay
        self.update_every = configure.update_every
        self.update_ratio = configure.update_ratio
        self.total_episodes = configure.total_episodes
        self.max_timestamp = configure.max_timestamp
        self.use_linear_eps_decay = configure.use_linear_eps_decay
        self.use_double_dqn = configure.use_double_dqn
        self.use_dueling_dqn = configure.use_dueling_dqn
        self.script_path = configure.script_path
        self.model_path = configure.model_path
        self.model_filename = configure.model_filename
        self.score_path = configure.score_path
        self.score_csvname = configure.score_csvname
        self.score_imagename = configure.score_imagename
        self.mp4_path = configure.mp4_path
        self.mp4_filename = configure.mp4_filename
        self.start_time = time()


        # 동일한 결과 재현을 위한 SEED 값 설정
        random.seed(self.seed)

        # Dueling Q Deep Network 생성
        if self.use_dueling_dqn:
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, self.dropout, self.seed, self.fc1_size, self.fc2_size)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, self.dropout, self.seed, self.fc1_size, self.fc2_size)
        # Q Deep Network 생성
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, self.dropout, self.seed, self.fc1_size, self.fc2_size)
            self.qnetwork_target = QNetwork(state_size, action_size, self.dropout, self.seed, self.fc1_size, self.fc2_size)

        # Optimizer 생성 (Adam 사용)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

        # ReplayBuffer 생성
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.seed)

        # Step 변수 초기화
        self.t_step = 0

    # Action 수행 결과 저장
    def step(self, state, action, reward, next_state, done):
        # ReplayBuffer에 결과 저장
        self.memory.add(state, action, reward, next_state, done)

        # Step 업데이트
        self.t_step += 1

        # Update 조건에 해당되면 업데이트 수행
        if self.t_step % self.update_every == 0:
            # ReplayBuffer에 batch_size * update_ratio 이상의 데이터가 쌓여 있으면 업데이트를 수행함
            if len(self.memory) > self.batch_size * self.update_ratio:
                # 샘플링 개수만큼 ReplayBuffer에서 experiences를 가져옴
                experiences = self.memory.sample()
                # Train 수행, Discount factor는 gamma 사용
                self.train(experiences, self.gamma)

    # Epsilon greedy를 사용한 Q값 선택
    def sample_action(self, state, eps):
        # State 가져오기
        state = torch.from_numpy(state).float().unsqueeze(0)
        # 모델의 Eval 모드 활성화 : Dropout, Batchnorm 등 불필요 기능 비활성화
        self.qnetwork_local.eval()
        # 메모리 최적화 : no_grad 옵션 상태인 경우, autograd engine을 꺼서 메모리를 최적화
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # 현재 State에서 Forward 수행
        self.qnetwork_local.forward(state)

        # Epsilon-greedy 사용 (Exploration-Exploitation Dilemma)
        # Epsilon 값보다 크다면 최선의 선택을 함
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        # Epsilon 값보다 작거나 같다면 가능한 Action 범위에서 Random 으로 선택
        else:
            return random.choice(np.arange(self.action_size))

    # 최적의 Q값 선택
    def action(self, state):
        # State 가져오기
        state = torch.from_numpy(state).float().unsqueeze(0)
        # 모델의 Eval 모드 활성화 : Dropout, Batchnorm 등 불필요 기능 비활성화
        self.qnetwork_local.eval()
        # 메모리 최적화 : no_grad 옵션 상태인 경우, autograd engine을 꺼서 메모리를 최적화
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # 현재 State에서 Forward 수행
        self.qnetwork_local.forward(state)

        # 최선의 선택을 함
        return np.argmax(action_values.cpu().data.numpy())

    # 모델 학습 함수
    def train(self, experiences, gamma):
        # 가져온 experiences를 사용하기 좋게 분리함
        states, actions, rewards, next_states, dones = experiences

        ### Algorithm 선택 - DQN 종류 ###
        # Double DQN
        if self.use_double_dqn:
            # 현재 학습중인 Neural network에서 maxQ 인 index 추출
            argmax_Q = self.qnetwork_local(states).max(1)[1].unsqueeze(1)
            # 추출한 index 값을 사용하여 Target network 값 확인
            q_targets_next = self.qnetwork_target(next_states).gather(1, argmax_Q)
        # Normal DQN
        else:
            # Target model에서 maxQ 인 index 값을 사용하여 확인
            q_targets_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)

        # 구해진 Q Target값 계산 (Y hat)
        # 마지막(dones=1) 인 경우는 rewards만 계산함
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # 실제 값 계산 (Y)
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # 손실 계산, MSE Loss 사용
        loss = F.mse_loss(q_expected, q_targets)

        # 값 보정
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 타겟 모델을 현재 모델 그대로 사용하지 않고, 낮은 비율만큼 유지시켜 줌
        # Target 모델 Soft update 수행 (tau * 현재 모델 + (1-tau) * 타겟 모델로 타겟 모델을 갱신)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    # 모델 Soft update 함수수
    def soft_update(self, local_model, target_model, tau):
        # target_param과 local_param을 가져옴
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # tau 만큼 보정하여 target_param에 할당
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    # 모델 Load 함수
    def load_model(self, model_filename):
        # 모델 파일은 필수, 없을 경우 종료
        if not os.path.isfile(model_filename):
            print('\nModel data not found : {}\n'.format(model_filename))
            exit()

        latest_model = torch.load(model_filename)
        self.qnetwork_local.load_state_dict(latest_model)

    # 모델 Save 함수
    def save_model(self, model_filename):
        # Replay 모드가 아니라면 마지막 모델 저장
        if not REPLAY:
            print('\nModel saved in {}\n'.format(model_filename))
            torch.save(self.qnetwork_local.state_dict(), model_filename)

    # 모델 Train 함수
    def train_model(self, env, eps, total_episodes, video=False):
        # 점수 저장용 리스트 생성
        scores = []
        avg_scores = []
        timestamps = []
        landed_timestamp = 0

        # 최근 점수 저장용 리스트 생성
        latest_scores = collections.deque(maxlen=100)

        for current_episode in range(1, total_episodes + 1):
            # Environment 초기화 (GYM 버전에 맞게 처리)
            if self.gym_old:
                state = env.reset()
            else:
                state, _info = env.reset()
            score = 0
            done = False

            for timestamp in range(self.max_timestamp):
                # Epsilon greedy를 사용하여 action 선택
                action = self.sample_action(state, eps)

                # Record 모드라면 프레임 기록
                if self.record:
                    video.append_data(env.render())

                # 정해진 Action 수행 (GYM 버전에 맞게 처리)
                if self.gym_old:
                    next_state, reward, done, _info = env.step(action)
                else:
                    next_state, reward, terminated, truncated, _info = env.step(action)
                    done = terminated or truncated

                # Replay 모드가 아니라면 수행한 Action 의 결과를 ReplayBuffer에 저장 및 Train 수행
                if not self.replay:
                    ### 학습 튜닝 ###
                    # LunarLander인 경우는 시간에 대한 벌점을 추가 (-0.5점)
                    if self.gym == 'LunarLander-v2':
                        self.step(state, action, reward-0.5, next_state, done)
                    else:
                        self.step(state, action, reward, next_state, done)

                # 다음번 State 할당
                state = next_state
                # Score 합산
                score += reward

                # Final state인 경우는 중단
                if done:
                    break

            # 최근 100개 점수 저장
            latest_scores.append(score)
            # 현재 점수 저장
            scores.append(score)
            avg_scores.append(np.mean(latest_scores))
            timestamps.append(timestamp)

            # Replay 모드가 아니라면 Epsilon 처리
            if not self.replay:
                # Epsilon 감소방식 선정
                if self.use_linear_eps_decay:
                    # Linear 하게 Epsilon 값 감소
                    eps = max(self.eps_end, self.eps_start - (1 - self.eps_decay)*(current_episode/2))
                else:
                    # 정해진 감소 비율만큼 감소 (Linear 보다 이게 더 잘되는듯)
                    eps = max(self.eps_end, self.eps_decay * eps)

            # Carriage return(\r)후 작성하고, end=""로 주면 같은 라인이 계속 업데이트되는 효과가 있음
            if current_episode % 100 != 0:
                print('\rEpisode {:4d}, Epsilon {:6.2f}%, Timestamp: {:4d}, Return: {:8.2f}, Average Score: {:7.2f}'.format(current_episode, eps*100, timestamp, score, np.mean(latest_scores)), end="")
            # 100개 단위로는 새로운 줄로 변경해서 완전히 기록함
            else:
                print('\rEpisode {:4d}, Epsilon {:6.2f}%, Timestamp: {:4d}, Return: {:8.2f}, Average Score: {:7.2f}'.format(current_episode, eps*100, timestamp, score, np.mean(latest_scores)))

            # 목표 Score에 도달하면 중단 (미사용)
            if np.mean(latest_scores) >= self.target_return and not self.replay and False:
                print('\nEnvironment solved in {:4d} episodes!, Average Score: {:7.2f}'.format(current_episode - 100, np.mean(latest_scores)))
                break

        # 소요시간 출력
        print('Total elapsed time: {}s'.format(int(time() - self.start_time)))

        # 점수 리턴
        self.df = pd.DataFrame({'Score': scores, 'Average Score': avg_scores, 'Timestamp': timestamps, 'Solved Requirement': self.target_return, 'Stretch Goal': self.stretch_goal})
        return self.df

    # 모델 Test 함수
    def test_model(self, env, eps, total_episodes, video):
        return self.train_model(env, eps, total_episodes, video)

    # 결과 저장 및 시각화
    def plot_results(self):
        if self.score_csvname:
            self.df.to_csv(self.score_csvname)

        fig = plt.figure(dpi=150)
        plt.tight_layout()
        ax = fig.add_subplot(1, 1, 1)
        ax.text(x=0.98, y=0.20, transform=ax.transAxes, s="Buffer: {}, Batch: {}".format(self.batch_size, self.buffer_size), fontsize=9, verticalalignment='top', horizontalalignment='right')
        ax.text(x=0.98, y=0.16, transform=ax.transAxes, s="EPS: (S){}/(E){}/(D){}".format(self.eps_start, self.eps_end, self.eps_decay), fontsize=9, verticalalignment='top', horizontalalignment='right')
        ax.text(x=0.98, y=0.12, transform=ax.transAxes, s="LR(Adam): {}, UR(TAU): {}, Dropout: {}".format(self.learning_rate, self.tau, self.qnetwork_local.dropout), fontsize=9, verticalalignment='top', horizontalalignment='right')
        ax.text(x=0.98, y=0.08, transform=ax.transAxes, s="# of HL: {}, # of Node: {}, Gamma: {}".format(2, self.qnetwork_local.fc1_size, self.gamma), fontsize=9, verticalalignment='top', horizontalalignment='right')
        ax.text(x=0.98, y=0.04, transform=ax.transAxes, s="GYM: {}, SEED: {}, WIND: {}".format(self.gym, self.seed, self.wind), fontsize=9, verticalalignment='top', horizontalalignment='right')
        plt.plot(self.df.index.values+1, 'Score', data=self.df, marker='', color='dodgerblue', linewidth=0.5, label='Score')
        plt.plot(self.df.index.values+1, 'Average Score', data=self.df, marker='', color='orange', linewidth=2, linestyle='dashed', label='Average Score')
        plt.plot(self.df.index.values+1, 'Solved Requirement', data=self.df, marker='', color='navy', linewidth=1.5, linestyle='dashed', label='Solved Requirement')
        plt.plot(self.df.index.values+1, 'Stretch Goal', data=self.df, marker='', color='red', linewidth=1.5, linestyle='dashed', label='Stretch Goal')
        plt.legend(loc='lower left', fontsize=9)
        plt.ylim([-1000, 300])
        if self.score_imagename:
            plt.savefig(self.score_imagename)

        #plt.show()
        plt.close()



# Dict를 .으로 Access 가능하도록 처리
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



# 메인 함수
def main():
    # DQN_TYPE 처리
    if USE_DOUBLE_DQN and USE_DUELING_DQN:
        dqn_type = 'DoubleDueling'
    elif USE_DOUBLE_DQN:
        dqn_type = 'Double'
    elif USE_DUELING_DQN:
        dqn_type = 'Dueling'
    else:
        dqn_type = 'DQN'

    # WIND 처리
    if WIND:
        wind = 'WIND'
    else:
        wind = 'NOWIND'

    # 설정값 Dictionary로 변환 (Loop 처리 용이하도록)
    CONFIGURE = {
        'dqn_type':dqn_type,
        'wind_type':wind,
        'gym_old':GYM_OLD,
        'replay':REPLAY,
        'record':RECORD,
        'gym':GYM,
        'wind':WIND,
        'seed':SEED,
        'target_return':TARGET_RETURN,
        'stretch_goal':STRETCH_GOAL,
        'fc1_size':FC1_SIZE,
        'fc2_size':FC2_SIZE,
        'buffer_size':BUFFER_SIZE,
        'batch_size':BATCH_SIZE,
        'dropout':DROPOUT,
        'gamma':GAMMA,
        'tau':TAU,
        'learning_rate':LEARNING_RATE,
        'eps_start':EPS_START,
        'eps_end':EPS_END,
        'eps_decay':EPS_DECAY,
        'update_every':UPDATE_EVERY,
        'update_ratio':UPDATE_RATIO,
        'total_episodes':TOTAL_EPISODES,
        'max_timestamp':MAX_TIMESTAMP,
        'use_linear_eps_decay':USE_LINEAR_EPS_DECAY,
        'use_double_dqn':USE_DOUBLE_DQN,
        'use_dueling_dqn':USE_DUELING_DQN,
        'script_path':SCRIPT_PATH,
        'model_path':MODEL_PATH,
        'score_path':SCORE_PATH,
        'mp4_path':MP4_PATH
    }
    LOOP_CONFIGURE = dotdict(CONFIGURE)

    # 테스트 모드이면 기본값으로만 수행
    if TEST:
        run(LOOP_CONFIGURE, 'FC1_SIZE', [FC1_SIZE])
    # 테스트 모드가 아니라면 수행
    else:
        # 설정값을 변경하면서 수행
        # WIND는 전체에 대해 수행함
        LOOP = {
            'WIND':['WIND', 'NOWIND'],
            'FC1_SIZE':[4, 32, 64, 128, 256, 512, 1024],
            'BUFFER_SIZE':[1000, 10000, 50000, 100000],
            'BATCH_SIZE':[4, 32, 64, 128],
            'DROPOUT':[0.0, 0.2, 0.4, 0.6, 0.8],
            'GAMMA':[0.8, 0.9, 0.99],
            'TAU':[0.1, 0.01, 0.001],
            'LEARNING_RATE':[0.5, 0.05, 0.005, 0.0005],
            'EPS_START':[0.9, 0.5, 0.1],
            'EPS_DECAY':[0.9, 0.99, 0.995],
            'DQN_TYPE':['DQN', 'Double', 'Dueling','DoubleDueling']
        }

        # WIND 여부에 따라서는 모두 수행함
        for w in LOOP['WIND']:
            # WIND 값 변경
            CONFIGURE['wind_type'] = w
            if CONFIGURE['wind_type'] == 'WIND':
                CONFIGURE['wind'] = True
            else:
                CONFIGURE['wind'] = False

            # 그 외 항목에 대해서는 각 항목별로만 변경하면서 수행
            for key, value in LOOP.items():
                # WIND는 제외
                if key == 'WIND':
                    continue

                # 설정값을 기본값으로 초기화
                LOOP_CONFIGURE = dotdict(CONFIGURE)
                # 현재 Key/Value에 대해 수행
                run(LOOP_CONFIGURE, key, value)



# 모델정보 전처리 함수
def run(CONFIGURE, type, loop):
    # 입력받은 테스트 항목(loop) 전체에 대해 반복 수행
    for item in loop:
        # 해당 값 변경
        if type == 'FC1_SIZE':
            CONFIGURE.fc1_size = item
            CONFIGURE.fc2_size = item
        elif type == 'BUFFER_SIZE':
            CONFIGURE.buffer_size = item
        elif type == 'BATCH_SIZE':
            CONFIGURE.batch_size = item
        elif type == 'DROPOUT':
            CONFIGURE.dropout = item
        elif type == 'GAMMA':
            CONFIGURE.gamma = item
        elif type == 'TAU':
            CONFIGURE.tau = item
        elif type == 'LEARNING_RATE':
            CONFIGURE.learning_rate = item
        elif type == 'EPS_START':
            CONFIGURE.eps_start = item
        elif type == 'EPS_DECAY':
            CONFIGURE.eps_decay = item
        elif type == 'DQN_TYPE':
            CONFIGURE.dqn_type = item
            if CONFIGURE.dqn_type == 'DQN':
                CONFIGURE.use_double_dqn = False
                CONFIGURE.use_dueling_dqn = False
            elif CONFIGURE.dqn_type == 'Double':
                CONFIGURE.use_double_dqn = True
                CONFIGURE.use_dueling_dqn = False
            elif CONFIGURE.dqn_type == 'Dueling':
                CONFIGURE.use_double_dqn = False
                CONFIGURE.use_dueling_dqn = True
            elif CONFIGURE.dqn_type == 'DoubleDueling':
                CONFIGURE.use_double_dqn = True
                CONFIGURE.use_dueling_dqn = True

        # 현재 수행 모델 출력
        print("> {}: {}".format(type, item))
        # 모델 수행
        run_model(CONFIGURE)
    


# 실제 모델정보 처리 함수
def run_model(CONFIGURE):
    # 저장정보 변경
    CONFIGURE.mode = 'TEST' if CONFIGURE.replay else 'TRAIN'
    CONFIGURE.model_filename = os.path.join(CONFIGURE.model_path, "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.zip".format(
        CONFIGURE.gym, CONFIGURE.mode, CONFIGURE.wind_type, CONFIGURE.fc1_size, CONFIGURE.buffer_size, CONFIGURE.batch_size, CONFIGURE.dropout, CONFIGURE.gamma, CONFIGURE.tau, CONFIGURE.learning_rate, CONFIGURE.eps_start, CONFIGURE.eps_decay, CONFIGURE.dqn_type))
    CONFIGURE.score_csvname = os.path.join(CONFIGURE.score_path, "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
        CONFIGURE.gym, CONFIGURE.mode, CONFIGURE.wind_type, CONFIGURE.fc1_size, CONFIGURE.buffer_size, CONFIGURE.batch_size, CONFIGURE.dropout, CONFIGURE.gamma, CONFIGURE.tau, CONFIGURE.learning_rate, CONFIGURE.eps_start, CONFIGURE.eps_decay, CONFIGURE.dqn_type))
    CONFIGURE.score_imagename = os.path.join(CONFIGURE.score_path, "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(
        CONFIGURE.gym, CONFIGURE.mode, CONFIGURE.wind_type, CONFIGURE.fc1_size, CONFIGURE.buffer_size, CONFIGURE.batch_size, CONFIGURE.dropout, CONFIGURE.gamma, CONFIGURE.tau, CONFIGURE.learning_rate, CONFIGURE.eps_start, CONFIGURE.eps_decay, CONFIGURE.dqn_type))
    CONFIGURE.mp4_filename = os.path.join(CONFIGURE.mp4_path, "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.mp4".format(
        CONFIGURE.gym, CONFIGURE.mode, CONFIGURE.wind_type, CONFIGURE.fc1_size, CONFIGURE.buffer_size, CONFIGURE.batch_size, CONFIGURE.dropout, CONFIGURE.gamma, CONFIGURE.tau, CONFIGURE.learning_rate, CONFIGURE.eps_start, CONFIGURE.eps_decay, CONFIGURE.dqn_type))

    # GYM 생성
    # Replay 모드일 경우
    if CONFIGURE.replay:
        # video 로컬변수 할당
        video = None
        # Record 모드일 경우, rgb_array 모드로 생성
        if CONFIGURE.record:
            env = gym.make(CONFIGURE.gym, enable_wind=CONFIGURE.wind, render_mode='rgb_array')
            video = imageio.get_writer(CONFIGURE.mp4_filename, fps=30)
            total_episodes = 5
        # 아니라면 human 모드로 생성
        else:
            # Platform이 Windows 일 때만 human으로 수행하여 UI 표시
            if sys.platform == 'win32':
                env = gym.make(CONFIGURE.gym, enable_wind=CONFIGURE.wind, render_mode='human')
            else:
                env = gym.make(CONFIGURE.gym, enable_wind=CONFIGURE.wind)
            total_episodes = 100
        # 표시할 Epsilon은 0%로 설정 (Deterministic policy)
        eps = 0.0
    # Train 모드일 경우 ascii 모드로 생성
    else:
        env = gym.make(CONFIGURE.gym, enable_wind=CONFIGURE.wind)
        # Epsilon-greedy 초기값 설정 (eps = EPS_START)
        eps = CONFIGURE.eps_start
        total_episodes = CONFIGURE.total_episodes

    # OLD 버전일 경우 SEED 설정
    if CONFIGURE.gym_old:
        print('Running on old GYM Environment...')
        env.seed(CONFIGURE.seed)
    else:
        print('Running on new GYM Environment...')
        env.reset(seed=CONFIGURE.seed)

    # GYM Environment의 state 개수 확인
    state_size = env.observation_space.shape[0]
    # GYM Environment의 action 개수 확인 (discrete)
    action_size = env.action_space.n

    # Agent 생성
    agent = Agent(state_size=state_size,
                  action_size=action_size,
                  configure=CONFIGURE)

    # 모델 Replay 모드라면 저장된 모델을 불러와서 적용함
    if CONFIGURE.replay:
        agent.load_model(CONFIGURE.model_filename.replace('TEST', 'TRAIN'))
        agent.test_model(env, eps, total_episodes, video)
    else:
        agent.train_model(env, eps, total_episodes)
        agent.save_model(CONFIGURE.model_filename)

    # 결과 저장
    agent.plot_results()



# Entrypoint
if __name__ == '__main__':
    main()
