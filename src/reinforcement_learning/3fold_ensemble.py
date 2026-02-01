import json
import numpy as np
import random
import torch
import torch.nn as nn
import os
from collections import deque

# =========================================================
# [Part 1] 하위 에이전트 및 IQN 네트워크 (그대로 유지)
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

class IQN_Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_quantiles=32):
        super(IQN_Network, self).__init__()
        self.input_dim = input_dim; self.output_dim = output_dim; self.n_quantiles = n_quantiles
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
# [Part 2] 3-Fold IQN 앙상블 에이전트 (최적화됨)
# =========================================================
class IQN3FoldEnsembleAgent:
    def __init__(self, n_arms=2):
        self.n_arms = n_arms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.action_dim = 3
        self.state_dim = 12
        
        # 1. 모델 로딩 (한 번만 수행)
        self.models = []
        model_files = ['iqn_fold_0.pth', 'iqn_fold_1.pth', 'iqn_fold_2.pth']
        
        print(f"🔄 Loading Models on {self.device}...")
        for f in model_files:
            if os.path.exists(f):
                model = IQN_Network(self.state_dim, self.action_dim).to(self.device)
                # weights_only=True 옵션을 쓰면 경고가 사라질 수 있으나, 호환성을 위해 유지
                model.load_state_dict(torch.load(f, map_location=self.device))
                model.eval()
                self.models.append(model)
                print(f"  ✅ Loaded: {f}")
            else:
                print(f"  ⚠️ Missing: {f}")
        
        # 2. 내부 변수 초기화
        self.reset_episode()

    def reset_episode(self):
        """에피소드마다 호출: 하위 에이전트와 기억 초기화"""
        self.sub_agents = [StockAgent(self.n_arms), HeonAgent(self.n_arms), AdvancedAgent(self.n_arms)]
        self.history = deque(maxlen=6)
        for _ in range(6): self.history.append((0,0))
        self.last_decisions = []
        self.last_action = 0

    def get_state(self):
        flat = []
        for a, r in self.history: flat.extend([a, r])
        return torch.FloatTensor(flat).to(self.device)

    def select_arm(self):
        # 하위 에이전트 선택
        act_stock = self.sub_agents[0].select_arm()
        act_heon = self.sub_agents[1].select_arm()
        act_adv = self.sub_agents[2].select_arm()
        self.last_decisions = [act_stock, act_heon, act_adv]
        
        # 앙상블 추론 (Soft Voting)
        if not self.models:
            meta_action = random.randint(0, 2)
        else:
            state = self.get_state()
            with torch.no_grad():
                state_tensor = state.unsqueeze(0)
                avg_q = torch.zeros(1, self.action_dim).to(self.device)
                
                for model in self.models:
                    quantiles, _ = model(state_tensor)
                    q_values = quantiles.mean(dim=1)
                    avg_q += q_values
                
                meta_action = avg_q.argmax().item()
        
        self.last_action = meta_action
        return self.last_decisions[meta_action]

    def update(self, arm, reward):
        # 하위 에이전트 업데이트
        for agent in self.sub_agents:
            if hasattr(agent, 'update'): agent.update(arm, reward)
        
        # History 업데이트
        self.history.append((arm, reward))

# =========================================================
# [Part 3] 환경 및 메인 평가 (속도 개선됨)
# =========================================================
class NonStationaryEnvironment:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.levels = json.load(f)
        self.levels.sort(key=lambda x: x['start_trial'])
        if self.levels: self.max_trial = self.levels[-1]['end_trial'] + 1
        else: self.max_trial = 2000

    def get_reward_probabilities(self, t):
        for level_data in self.levels:
            if level_data['start_trial'] <= t <= level_data['end_trial']:
                return [level_data['p0'], level_data['p1']]
        return [0.5, 0.5]

    def get_reward(self, arm, t):
        probs = self.get_reward_probabilities(t)
        if arm < len(probs): return 1 if random.random() < probs[arm] else 0
        return 0

def run_evaluation():
    json_files = ['rwd_seq_example_01.json', 'rwd_seq_example_02.json', 'rwd_seq_example_03.json']
    N_EPISODES = 100
    
    print(f"\n🚀 Evaluating 3-Fold IQN Ensemble (100 Episodes each)...")
    
    # [수정] 에이전트를 루프 밖에서 한 번만 생성 (모델 로딩 1회)
    agent = IQN3FoldEnsembleAgent()
    if not agent.models:
        print("❌ 모델이 없습니다. train_3fold_iqn.py를 먼저 실행하세요.")
        return

    print(f"{'File Name':<25} | {'Avg Score':<10} | {'Max':<6} | {'Min':<6}")
    print("-" * 60)
    
    for fname in json_files:
        if not os.path.exists(fname):
            print(f"{fname:<25} | FILE NOT FOUND")
            continue
            
        scores = []
        for ep in range(N_EPISODES):
            env = NonStationaryEnvironment(fname)
            
            # [수정] 모델을 다시 로드하지 않고 메모리만 초기화
            agent.reset_episode() 
            
            total_r = 0
            for t in range(env.max_trial):
                action = agent.select_arm()
                reward = env.get_reward(action, t)
                agent.update(action, reward)
                total_r += reward
            scores.append(total_r)
            
        avg = sum(scores) / len(scores)
        print(f"{fname:<25} | {avg:<10.1f} | {max(scores):<6} | {min(scores):<6}")

    print("-" * 60)
    print("✅ Evaluation Complete!")

if __name__ == "__main__":
    run_evaluation()
