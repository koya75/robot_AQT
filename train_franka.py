import argparse
import logging
from pathlib import Path
import os

from isaacgym.torch_utils import to_torch

import torch.optim as optim
import torch
import repos.pfrl.pfrl as pf
from repos.pfrl.pfrl import experiments
from repos.pfrl.pfrl.agents import double_dqn
from repos.pfrl.pfrl.agents import ppo

import numpy as np

import requests
import datetime
import pytz
from make_urdf import URDF
from demo_module import DEMO
import numpy
import matplotlib.pyplot as plt

##############################################################
#                           env
##############################################################
def to_tensor(x):
    if isinstance(x, list):
        if isinstance(x[0], list):
            x_tensor = []
            for y in x:
                x_tensor.append(to_tensor(y))
        else:
            x_tensor = x
        return torch.stack(x_tensor, dim=0)
    else:
        return x

# def batch_states(states, device, phi):
#     return to_tensor(states)

def line_notify(message):
    line_notify_token = ''
    line_notify_api = 'https://notify-api.line.me/api/notify'
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token} 
    requests.post(line_notify_api, data=payload, headers=headers)

def main():
    urdf = URDF()
    urdf.create_urdf()

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--isaacgym-assets-dir", type=str, default="../isaacgym/assets")
    parser.add_argument("--item-urdf-dir", type=str, default="./urdf")
    parser.add_argument(
        "--model",
        help="choose model",
        type=str,
        choices=[
            "proto1",
            "proto2",
            "mask_single_value",
            "mask_single_policy",
            "mask_double",
            "target_mask_double",
            "dqn",
            "Action_Q_Transformer_model",
        ],
    )
    parser.add_argument(
        "--use-lstm", action="store_true", default=False, help="use lstm"
    )
    parser.add_argument("--steps", type=int, default=10 ** 7)
    parser.add_argument("--step_offset", type=int, default=0)
    parser.add_argument("--update-batch-interval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument(
        "--lambd", type=float, default=0.95, help="Lambda-return factor [0, 1]"
    )
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-batch-interval", type=int, default=10 ** 4)
    parser.add_argument("--eval-n-runs", type=int, default=128)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument(
        "--max-grad-norm", type=float, default=40, help="value loss coefficient"
    )
    """parser.add_argument('--gpu-ids', type=str, default='0,1',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')"""
    parser.add_argument(
        "--gpu",
        "-g",
        type=int,
        default=0,
        help="GPU ID (negative value indicates CPU)",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument(
        "--output-sensor-images-dir",
        type=str,
        default=None,
        help="Output sensor images directory. Image files are updated per step",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=1,
        help="Number of items on tray.",
    )
    parser.add_argument(
        "--item-names", nargs="+", default=None, help=["List of item names."]
    )
    parser.add_argument("--descentstep", type=int, default=12)
    
    parser.add_argument(
        "--target",
        type=str,
        default="item21",
        help="target-item",
    )
    parser.add_argument("--drop", action="store_true", default=False, help="use drop")
    parser.add_argument("--hand", action="store_true", default=False, help="use handcamera")
    parser.add_argument(
        "--mode",
        help="choose mode",
        type=str,
        default="normal",
        choices=[
            "normal",
            "hard",
            "veryhard",
        ],
    )
    parser.add_argument(
        "--franka",
        help="choose mode",
        type=str,
        default="franka3",
        choices=[
            "franka3",
            "franka4",
            "franka5",
        ],
    )
    parser.add_argument(
        "--reward",
        help="choose mode",
        type=str,
        default="Linear",
        choices=[
            "Discrete",
            "Linear",
        ],
    )

    parser.set_defaults(use_lstm=False)
    args = parser.parse_args()
    
    logging.basicConfig(level=args.log_level)
    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))
    outdir = args.outdir

    height = 128
    width = 128
    action_repeat = 30
    item_asset_root = args.item_urdf_dir
    isaacgym_asset_root = args.isaacgym_assets_dir

    num_items = args.num_items
    item_names = args.item_names
    if item_names is None:
        item_names = sorted(list(Path(item_asset_root).glob("*.urdf")))
        item_names = [path.stem for path in item_names]
    
    if args.output_sensor_images_dir is not None:
        output_debug_images_dir = outdir + args.output_sensor_images_dir
    else:
        output_debug_images_dir = args.output_sensor_images_dir

    from envs.franka_grasping_env import FrankaGraspingEnv
    n_actions = 7

    if args.model == "mask_double":
        from agent.model_mask_double import ActorCritic#grasping_rl.

        model = ActorCritic(
            state_size=(height, width),
            action_size=envs.action_space.shape[0],
            critic_hidden_layers=[64],
            actor_hidden_layers=[64],
            init_type="xavier-uniform",
            seed=0,
            use_lstm=True,
        )
        agent_model = "ppo"
        discrete = False
    elif args.model == "target_mask_double":
        from agent.target_mask_double import ActorCritic#grasping_rl.

        model = ActorCritic(
            state_size=(height, width),
            action_size=envs.action_space.shape[0],
            critic_hidden_layers=[64],
            actor_hidden_layers=[64],
            init_type="xavier-uniform",
            seed=0,
            use_lstm=True,
            target = args.target,
        )
        agent_model = "ppo"
        discrete = False
    elif args.model == "dqn":
        from agent.dqn_model import DQN#grasping_rl.

        model = DQN(
            n_actions=n_actions,
        )
        agent_model = "dqn"
        discrete = True
    elif args.model == "Action_Q_Transformer_model":
        from agent.Action_Q_Transformer_model import TransDQN#grasping_rl.

        model = TransDQN(
            n_actions=n_actions,
        )
        agent_model = "dqn"
        discrete = True

    def make_batch_env(num_envs):
        env = FrankaGraspingEnv(
            num_envs=num_envs,
            height=height,
            width=width,
            discrete=discrete,
            image_type="color",
            item_asset_root=item_asset_root,
            isaacgym_asset_root=isaacgym_asset_root,
            num_items=num_items,
            item_names=item_names,
            use_viewer=args.render,
            action_repeat=action_repeat,
            output_debug_images_dir=output_debug_images_dir,
            device_id=args.gpu,
            n_actions = n_actions,
            descentstep = args.descentstep,
            target = args.target,
        )
        return env

    envs = make_batch_env(num_envs=args.num_envs)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    update_step_interval = args.num_envs * envs.max_episode_length * args.update_batch_interval

    print(envs.action_space.sample)
    if agent_model == "dqn":
        agent = double_dqn.DoubleDQN(
            model,
            optimizer,
            replay_buffer=pf.replay_buffers.ReplayBuffer(capacity=8000), #4000, 6000, 20000
            gamma=args.gamma,
            explorer=pf.explorers.LinearDecayEpsilonGreedy( # 探索(ε-greedy)
                start_epsilon=1.0, end_epsilon=0.02 ,decay_steps=1000000, random_action_func=envs.action_space.sample),
            replay_start_size=6000,# 1000, 3000, 15000
            gpu=args.gpu,
            update_interval=update_step_interval,
            target_update_interval=update_step_interval*100,
            minibatch_size=(args.descentstep * args.num_envs),
            phi=lambda x: x.to(torch.float32, copy=False),
            recurrent=args.use_lstm,
        )
    elif agent_model == "ppo":
        agent = ppo.PPO(
        model,
        optimizer,
        gamma=args.gamma,
        lambd=args.lambd,
        gpu=args.gpu,
        update_interval=update_step_interval,
        max_grad_norm=args.max_grad_norm,
        minibatch_size=(args.descentstep * args.num_envs),
        epochs=args.epochs,
        recurrent=args.use_lstm,
        )

    if args.load:
        agent.load(args.load)

    if args.demo:
        demo_module = DEMO
        demo_module.create_dir(agent_model, args.outdir, args.num_envs)

        with agent.eval_mode():
            obs = envs.reset()
            obss = obs
            step = 0
            batch_size = envs.num_envs
            camera_ani_all = []
            enc_ani_all = []
            dec_ani_all = []
            ani_camera = []
            ani_enc = []
            ani_dec = []
            while True:
                if agent_model == "dqn":
                    camera_ani = []
                    enc_ani = []
                    dec_ani = []

                    episode = step // envs.max_episode_length

                    actions = agent.batch_act(obss)#args.demo, 
                    with open(Path(args.outdir).joinpath("q_values.txt"), "a") as place:
                        np.savetxt(place, np.transpose(agent.batch_argmax), fmt='%d', delimiter="\n", newline=",", footer="\nkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk\n")
                        np.savetxt(place, agent.batch_value, fmt='%.4f', delimiter=",", newline="\n")
                    #print(actions[0], actions[1], actions[2], actions[3], actions[4], actions[5], actions[6], actions[7], actions[8], actions[9], actions[10], actions[11])

                    camera_ani, enc_ani, dec_ani = demo_module.make_image(agent, camera_ani, enc_ani, dec_ani, camera_ani_all, enc_ani_all, dec_ani_all, batch_size, n_actions)

                    obss, rs, dones, infos = envs.step(actions)

                    
                    camera_ani_all = camera_ani if camera_ani_all == [] else np.concatenate([camera_ani_all,camera_ani], 1)
                    enc_ani_all = enc_ani if enc_ani_all == [] else np.concatenate([enc_ani_all, enc_ani], 1)
                    dec_ani_all = dec_ani if dec_ani_all == [] else np.concatenate([dec_ani_all,dec_ani], 2)

                    if step % envs.max_episode_length == 0 and step > 0:
                        demo_module.save_image(camera_ani_all, enc_ani_all, dec_ani_all, step, batch_size, episode, n_actions)

                    step +=1

                    if step % envs.max_episode_length == 0:
                        print(rs)
                        rs = list(rs)
                        now = datetime.datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%m%d%H%M%S")
                        """score = "rsj/" + args.model +"_"+ str(now) + '_score.txt'
                        np.savetxt(score, rs, delimiter=",")"""
                        sam = sum(rs)
                        age = np.mean(rs)
                        print("合計：", sam, "平均値：",age)
                    
                    resets = np.zeros(args.num_envs, dtype=bool)

                    # Agent observes the consequences
                    agent.batch_observe(obss, rs, dones, resets)

                    # Make mask. 0 if done/reset, 1 if pass
                    end = np.logical_or(resets, dones)
                    not_end = np.logical_not(end)

                    obs = envs.reset(not_end)
                    obss = obs

                elif agent_model == "ppo":
                    actions = agent.batch_act(obss, args.demo)
                    obss, rs, dones, infos = envs.step(actions)
                    
                    if args.model != "noattention":
                        demo_module.write_mask(envs, agent, step, args.model)

                    step +=1
                    if step % 12 == 0:
                        print(rs)
                        rs = list(rs)
                        now = datetime.datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%m%d%H%M%S")
                        """score = "rsj/" + args.model +"_"+ str(now) + '_score.txt'
                        np.savetxt(score, rs, delimiter=",")"""
                        sam = sum(rs)
                        age = np.mean(rs)
                        print("合計：", sam, "平均値：",age)
                    
                    resets = np.zeros(args.num_envs, dtype=bool)

                    # Agent observes the consequences
                    agent.batch_observe(obss, rs, dones, resets)

                    # Make mask. 0 if done/reset, 1 if pass
                    end = np.logical_or(resets, dones)
                    not_end = np.logical_not(end)

                    obss = envs.reset(not_end)

    else:

        step_hooks = []

        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = value

        step_hooks.append(
            experiments.LinearInterpolationHook(args.steps, args.lr, 0, lr_setter)
        )
        eval_interval =  envs.max_episode_length * envs.num_envs * args.eval_batch_interval
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=envs,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=eval_interval,
            outdir=args.outdir,
            step_offset=args.step_offset,
            save_best_so_far_agent=True,
            log_interval=update_step_interval,
            step_hooks=step_hooks,
            checkpoint_freq=1000000,
            use_tensorboard=False,
            agent_model=agent_model,
        )

    message = "finish_"+ str(args.gpu)
    line_notify(message)

if __name__ == "__main__":
    main()
