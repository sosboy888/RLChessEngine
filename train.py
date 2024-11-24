def train(env, model, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Predict Q-values
            q_values = model(state_tensor)

            # Choose action (ε-greedy strategy)
            if np.random.rand() < epsilon:
                action = random.choice(env.legal_moves())
            else:
                action = torch.argmax(q_values).item()
            
            # Apply action
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Update Q-value
            optimizer.zero_grad()
            loss = criterion(q_values, reward)
            loss.backward()
            optimizer.step()

        print(f"Episode {episode}: Total Reward = {total_reward}")
