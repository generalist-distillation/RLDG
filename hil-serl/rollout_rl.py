#!/usr/bin/env python3
import datetime
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.utils.launcher import make_sac_pixel_agent
from serl_experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)

def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    agent: SACAgent = make_sac_pixel_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
    )
    agent: SACAgent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )
    ckpt = checkpoints.restore_checkpoint(
        FLAGS.checkpoint_path,
        agent.state,
        step=FLAGS.eval_checkpoint_step,
    )
    agent = agent.replace(state=ckpt)

    success_counter = 0
    transitions = []        
    for episode in range(FLAGS.eval_n_trajs):
        trajectory = []
        obs, info = env.reset()
        original_state_obs = info.pop("original_state_obs")
        done = False
        while not done:
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                argmax=False,
                seed=key,
            )
            actions = np.asarray(jax.device_get(actions))

            next_obs, reward, done, truncated, info = env.step(actions)
            next_original_state_obs = info.pop("original_state_obs")
            
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
                infos=info,
                original_state_obs=original_state_obs,
                next_original_state_obs=next_original_state_obs,
            )
            trajectory.append(copy.deepcopy(transition))

            obs = next_obs
            original_state_obs = next_original_state_obs
            if done:
                if reward:
                    transitions.extend(trajectory)
                    success_counter += 1
                print(reward)
                print(f"{success_counter}/{episode + 1}")
                trajectory = []
                
                
    buffer_path = os.path.join(FLAGS.checkpoint_path, f"rollout_data_{FLAGS.eval_checkpoint_step}")
    if not os.path.exists(buffer_path):
        os.makedirs(buffer_path)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(os.path.join(buffer_path, f"{success_counter}_successes_{uuid}.pkl"), "wb") as f:
        pkl.dump(transitions, f)
    print(f"saved {success_counter} successful trajectories")

if __name__ == "__main__":
    app.run(main)