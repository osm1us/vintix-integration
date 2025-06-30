import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """
    Буфер для хранения переходов (state, action, reward, next_state, done).
    Использует deque для эффективного добавления и удаления старых записей.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Добавляет переход в буфер."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Возвращает случайную выборку переходов размером batch_size."""
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.vstack(state), action, reward, np.vstack(next_state), done

    def __len__(self):
        """Возвращает текущий размер буфера."""
        return len(self.buffer)

class Actor(nn.Module):
    """
    Нейросеть "Исполнителя" (Actor).
    Принимает на вход состояние (observation) и выдает действие (action).
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, action_dim) # Слой для среднего значения действия
        self.log_std_layer = nn.Linear(256, action_dim) # Слой для логарифма станд. отклонения

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_layer(x)
        
        # Ограничиваем log_std, чтобы избежать слишком больших или малых значений
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)

        # Создаем нормальное распределение, из которого будем сэмплировать действие
        dist = Normal(mean, std)
        # Сэмплируем действие
        action = dist.rsample()
        # Применяем `tanh`, чтобы действие было в диапазоне [-1, 1]
        action_tanh = torch.tanh(action)
        
        # Вычисляем логарифм вероятности этого действия
        log_prob = dist.log_prob(action)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return self.max_action * action_tanh, log_prob

class Critic(nn.Module):
    """
    Нейросеть "Критика" (Critic).
    Принимает на вход состояние и действие, и выдает Q-значение (оценку "хорошести" действия).
    В SAC используется ДВЕ таких сети для стабильности.
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # --- Первая сеть критика ---
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # --- Вторая сеть критика ---
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        # Прямой проход для первой сети
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # Прямой проход для второй сети
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        """Возвращает Q-значение только от первой сети (нужно для обучения Actor)."""
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class SAC:
    """
    Класс, реализующий алгоритм Soft Actor-Critic (SAC).
    """
    def __init__(self, state_dim, action_dim, max_action, replay_buffer_capacity=1000000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"INFO: SAC-агент будет использовать устройство: {self.device}")

        # --- Гиперпараметры ---
        self.gamma = 0.99  # Дисконтирующий фактор
        self.tau = 0.005   # Коэффициент для "мягкого" обновления целевых сетей
        self.alpha = 0.2   # Коэффициент энтропии (можно сделать обучаемым)
        self.batch_size = 256
        self.learning_rate = 3e-4

        # --- Сети ---
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        
        # Копируем веса из основной сети критика в целевую
        self.critic_target.load_state_dict(self.critic.state_dict())

        # --- Оптимизаторы ---
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        # --- Буфер опыта ---
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        
        self.max_action = max_action

    def select_action(self, state):
        """Выбирает действие на основе текущего состояния."""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, _ = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def update(self, n_iterations=1):
        """Обновляет веса сетей на основе данных из буфера."""
        if len(self.replay_buffer) < self.batch_size:
            return

        for i in range(n_iterations):
            # 1. Сэмплируем батч из буфера
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

            # Конвертируем все в тензоры
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(np.vstack(action)).to(self.device)
            reward = torch.FloatTensor(np.vstack(reward)).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.FloatTensor(np.vstack(1 - np.array(done, dtype=np.float32))).to(self.device)
            
            # --- 2. Обучение Критика ---
            with torch.no_grad():
                # Получаем следующее действие и логарифм его вероятности от Actor
                next_action, log_prob = self.actor(next_state)
                
                # Вычисляем Q-значения для следующего состояния от целевого критика
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                
                # Вычитаем энтропию из Q-значения (ключевая идея SAC)
                target_q = target_q - self.alpha * log_prob
                
                # Формула Беллмана для вычисления целевого Q
                target_q = reward + (done * self.gamma * target_q)

            # Получаем текущие Q-значения
            current_q1, current_q2 = self.critic(state, action)
            
            # Считаем ошибку (MSE) между текущим и целевым Q
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            # Оптимизируем критика
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # --- 3. Обучение Актора ---
            # "Замораживаем" градиенты критика, чтобы обучать только актора
            for params in self.critic.parameters():
                params.requires_grad = False

            # Получаем действие и логарифм его вероятности
            new_action, log_prob = self.actor(state)
            
            # Получаем Q-значение для этого действия
            q1_new, q2_new = self.critic(state, new_action)
            q_new = torch.min(q1_new, q2_new)
            
            # Ошибка актора: мы хотим максимизировать Q-значение и энтропию
            actor_loss = (self.alpha * log_prob - q_new).mean()

            # Оптимизируем актора
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # "Размораживаем" градиенты критика
            for params in self.critic.parameters():
                params.requires_grad = True

            # --- 4. "Мягкое" обновление целевых сетей ---
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        """Сохраняет веса моделей."""
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.actor.state_dict(), filename + "_actor.pth")

    def load(self, filename):
        """Загружает веса моделей."""
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor.load_state_dict(torch.load(filename + "_actor.pth")) 