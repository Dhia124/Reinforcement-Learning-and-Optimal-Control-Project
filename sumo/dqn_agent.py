import numpy as np
import random
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from collections import deque
import time
import gym
import os
class DQNAgent:
    def __init__(self, state_size, action_size, num_episodes=1000,learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995, memory_size=2000):
        """
        Initialize Deep Q-Learning Agent.
        :param state_size: The size of the state space.
        :param action_size: The size of the action space.
        :param learning_rate: Learning rate for the neural network.
        :param discount_factor: Discount factor for future rewards.
        :param exploration_rate: Initial exploration rate.
        :param exploration_decay: Decay rate for the exploration.
        :param memory_size: Size of the memory buffer.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.model = self._build_model()
        self.num_episodes = num_episodes
        self.max_time_steps_per_episode = 500
        # Créez des sous-dossiers pour les fichiers d'entraînement et de test
        current_time = time.time()
        self.train_log_dir = f"logs/{current_time}/train"
        self.test_log_dir = f"logs/{current_time}/test"
        os.makedirs(self.train_log_dir)
        os.makedirs(self.test_log_dir)

        # Ajouter un callback TensorBoard
        # Ajouter un callback TensorBoard pour les fichiers d'entraînement
        self.tensorboard_train = TensorBoard(log_dir=self.train_log_dir)

        # Ajouter un callback TensorBoard pour les fichiers de test
        self.tensorboard_test = TensorBoard(log_dir=self.test_log_dir)
        print(f"Log directory: {self.train_log_dir}")
    def _build_model(self):
        """
        Build a neural network model.
        """
        model = Sequential()
        model.add(Dense(24, input_dim=4, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """
        Choose the action based on the current state.
        """
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size, episode, is_training=True):
        """
        Train the model using randomly sampled experiences from memory.
        """
        minibatch = random.sample(self.memory, batch_size)

        states = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target

            states.append(state[0])
            targets.append(target_f[0])

        states = np.array(states)
        targets = np.array(targets)

        if is_training:
            # Utilisez le callback TensorBoard approprié en fonction de is_training
            self.model.fit(states, targets, epochs=100, verbose=0, callbacks=[self.tensorboard_train])
        else:
            self.model.fit(states, targets, epochs=100, verbose=0, callbacks=[self.tensorboard_test])

        # Ajustez le taux d'exploration à la fin de chaque épisode
        if self.exploration_rate > 0.01:
            self.exploration_rate *= self.exploration_decay

        # Vous pouvez utiliser la variable 'episode' ici si nécessaire
        # par exemple, pour imprimer le numéro d'épisode à la fin de chaque épisode
        print(f"Épisode {episode} terminé.")

            
    def save(self, filename):
        """
        Save the trained model.
        """
        
        self.model.save(filename)

    def load(self, filename):
        """
        Load a trained model.
        """
        self.model.load_weights(filename)
    def train(self, batch_size=32):
        env = gym.make('CartPole-v1')

        """
        Entraînez le modèle en utilisant le nombre d'épisodes spécifié.
        """
        for episode in range(self.num_episodes):
            # Obtenez l'état initial de l'environnement
            state = env.reset()[0]  # Accédez à la première valeur du tuple
            print("Forme de l'état avant le remodelage :", state.shape)

            # Assurez-vous que self.state_size est défini sur la taille correcte de l'état
            self.state_size = 4  # Remplacez par la taille correcte de l'état

            # Convertissez l'état en une forme appropriée pour l'agent
            state = np.reshape(state, [1, self.state_size])
            print("Forme de l'état après le remodelage :", state.shape)

            for _ in range(self.max_time_steps_per_episode):
    # Choisissez une action en fonction de l'état actuel
                action = self.choose_action(state)

                # Exécutez l'action dans l'environnement
                step_result = env.step(action)
                print("Résultat de env.step :", step_result)  # Ajoutez cette ligne pour déboguer

                # Déballez les valeurs correctement
                next_state, reward, done, _ = step_result[:4]

                # Convertissez le prochain état en une forme appropriée
                next_state = np.reshape(next_state, [1, self.state_size])

                # Stockez l'expérience dans la mémoire de l'agent
                self.remember(state, action, reward, next_state, done)

                # Passez au prochain état
                state = next_state

                # Vérifiez si l'épisode est terminé
                if done:
                    break


            # Appel de la méthode replay de l'agent à la fin de chaque épisode
        self.replay(batch_size, episode, is_training=True)  # Déplacez cette ligne en dehors de la boucle d'épisodes

        # Sauvegarde du modèle après l'entraînement
        self.save("nom_du_modele.h5")



