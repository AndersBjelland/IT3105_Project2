from src import actor
from src import oht

if __name__ == '__main__':
    agent = actor.SFAgent(
        leaf_evaluation='rollout',
        encoder='normalized',
        policy='greedy',
        model_path='/Users/akselborgen/Downloads/oht6x6-v34',
        size=6,
        simulations=10,
        c=3)

    bsa = oht.BasicClientActor(agent, verbose=True)
    bsa.connect_to_server()
