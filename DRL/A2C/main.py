import agent
import policy
import gym

def main():
        
    # lets test our Agent!
    env = gym.make('CartPole-v0') 
    model = policy.RLModel(num_actions=env.action_space.n)

    Agent = agent.A2CAgent(model)
    #rewards_sum = Agent.test(env)
    print("Starts training...")
    rewards_history = Agent.train(env)
    print("Finished training, testing...")
    print("%d out of 200" % Agent.test(env,render=False)) # 200 out of 200
    print ("history: " , rewards_history)



if __name__ == "__main__":
    main()
