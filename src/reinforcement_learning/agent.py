import numpy as np
import torch
import torch.nn as nn
from collections import deque
import os
import random

# =========================================================
# [1] IQN 네트워크 정의 (불러오기 위해 필요)
# =========================================================
class IQN_Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_quantiles=32):
        super(IQN_Network, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_quantiles = n_quantiles
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.phi = nn.Sequential(nn.Linear(64, hidden_dim), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.pis = torch.FloatTensor([np.pi * i for i in range(1, 65)]).view(1, 1, 64).to(self.device)

    def forward(self, state, num_quantiles=None):
        if num_quantiles is None: num_quantiles = self.n_quantiles
        batch_size = state.shape[0]
        x = self.feature_layer(state)
        tau = torch.rand(batch_size, num_quantiles).to(self.device)
        tau_embed = torch.cos(tau.unsqueeze(-1) * self.pis)
        tau_embed = self.phi(tau_embed)
        x = x.unsqueeze(1)
        z = x * tau_embed
        quantiles = self.fc(z)
        return quantiles, tau

# =========================================================
# [2] 하위 에이전트들 (Sub-Agents)
# =========================================================
class AdvancedAgent: 
    def __init__(self, n_arms=2):
        self.n_arms = n_arms; self.q_values = np.ones(n_arms) * 0.5; self.tau, self.alpha = 0.1, 0.1
    def select_arm(self):
        pref = self.q_values - np.max(self.q_values)
        probs = np.exp(pref/self.tau) / np.sum(np.exp(pref/self.tau))
        return np.random.choice(self.n_arms, p=probs)
    def update(self, arm, reward): self.q_values[arm] += self.alpha * (reward - self.q_values[arm])

class HeonAgent: 
    def __init__(self, n_arms=2):
        self.n_arms = n_arms; self.win = 65; self.c = 0.8; self.hist = []
    def select_arm(self):
        if len(self.hist) < self.n_arms: return len(self.hist) % self.n_arms
        cw = self.hist[-self.win:]; cnts = np.zeros(self.n_arms); vals = np.zeros(self.n_arms)
        for a, r in cw: cnts[a]+=1; vals[a]+=r
        ucb = [vals[a]/cnts[a] + self.c*np.sqrt(np.log(len(cw))/cnts[a]) if cnts[a]>0 else 1e5 for a in range(self.n_arms)]
        return np.argmax(ucb)
    def update(self, a, r): self.hist.append((a, r))

class StockAgent: 
    def __init__(self, n_arms=2):
        self.n_arms = n_arms; self.last_a = None; self.last_r = None; self.loss_cnt = 0
    def select_arm(self):
        if self.last_a is None: return random.randint(0, self.n_arms-1)
        if self.last_r == 1: return self.last_a
        return 1-self.last_a if self.loss_cnt >= 2 else self.last_a
    def update(self, a, r):
        self.last_a = a; self.last_r = r; self.loss_cnt = self.loss_cnt + 1 if r == 0 else 0

# =========================================================
# [3] 메인 에이전트 (test_code.py 호환용 Wrapper)
# =========================================================
class BanditAgent:
    def __init__(self, n_arms=2):
        self.n_arms = n_arms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 하위 에이전트 초기화
        self.sub_agents = [StockAgent(n_arms), HeonAgent(n_arms), AdvancedAgent(n_arms)]
        self.action_dim = 3 # Stock, Heon, Adv
        self.state_dim = 12 # State 차원
        
        # 모델 로드 (3-Fold Ensemble)
        self.models = []
        model_files = ['iqn_fold_0.pth', 'iqn_fold_1.pth', 'iqn_fold_2.pth']
        
        print(f"🔄 [Agent] Loading 3-Fold Models on {self.device}...")
        for f in model_files:
            if os.path.exists(f):
                try:
                    model = IQN_Network(self.state_dim, self.action_dim).to(self.device)
                    # weights_only=False는 pickle 로딩 경고를 방지하기 위함일 수 있으나,
                    # 최신 버전에서는 weights_only=True 권장. 에러나면 False로 변경.
                    model.load_state_dict(torch.load(f, map_location=self.device))
                    model.eval()
                    self.models.append(model)
                    print(f"  ✅ Loaded: {f}")
                except Exception as e:
                    print(f"  ⚠️ Error loading {f}: {e}")
            else:
                print(f"  ⚠️ Missing: {f}")
        
        if not self.models:
            print("❌ No models loaded! Will behave randomly.")

        # 상태 관리 (History)
        self.history = deque(maxlen=6)
        for _ in range(6): self.history.append((0,0))
        self.last_decisions = []

    def _get_state(self):
        flat = []
        for a, r in self.history: flat.extend([a, r])
        return torch.FloatTensor(flat).to(self.device)

    def select_arm(self):
        # 1. 하위 에이전트들의 추천 수집
        act_stock = self.sub_agents[0].select_arm()
        act_heon = self.sub_agents[1].select_arm()
        act_adv = self.sub_agents[2].select_arm()
        self.last_decisions = [act_stock, act_heon, act_adv]
        
        # 2. 앙상블 모델로 최고의 전문가 선택 (Soft Voting)
        if not self.models:
            meta_action = random.randint(0, 2)
        else:
            state = self._get_state()
            with torch.no_grad():
                state_tensor = state.unsqueeze(0) # [1, 12]
                avg_q = torch.zeros(1, self.action_dim).to(self.device)
                
                # 3개 모델의 의견 종합
                for model in self.models:
                    quantiles, _ = model(state_tensor)
                    q_values = quantiles.mean(dim=1)
                    avg_q += q_values
                
                meta_action = avg_q.argmax().item()
        
        # 3. 최종 행동 결정
        final_action = self.last_decisions[meta_action]
        return final_action

    def update(self, arm, reward):
        # 1. 하위 에이전트 업데이트 (내부 상태 갱신)
        for agent in self.sub_agents:
            if hasattr(agent, 'update'):
                agent.update(arm, reward)
            elif hasattr(agent, 'update_reward'):
                agent.update_reward(arm, reward)
        
        # 2. 내 히스토리 업데이트 (다음 턴 판단을 위해)
        self.history.append((arm, reward))
