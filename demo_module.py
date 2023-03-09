from pathlib import Path
import os
import numpy as np
import cv2
from PIL import Image

def min_max(x, mins, maxs, axis=None):
    result = (x - mins)/(maxs - mins)
    return result

class DEMO:
    def create_dir(self, model, outdir, num_envs):
        if model == "dqn":
            self.gif_dir = Path(outdir).joinpath("gif_images")
            gif_input = Path(self.gif_dir).joinpath("gif_input")
            gif_enc = Path(self.gif_dir).joinpath("gif_enc")
            gif_dec = Path(self.gif_dir).joinpath("gif_dec")
            self.image_dir = Path(outdir).joinpath("images")
            dec_images = Path(self.image_dir).joinpath("dec_image")
            enc_images = Path(self.image_dir).joinpath("enc_image")
            image_input = Path(self.image_dir).joinpath("image_input")
            os.makedirs(self.gif_dir)
            os.makedirs(gif_input)
            os.makedirs(gif_enc)
            os.makedirs(gif_dec)
            os.makedirs(self.image_dir)
            os.makedirs(dec_images)
            os.makedirs(enc_images)
            os.makedirs(image_input)
            for idx in range(num_envs):
                dec_idx = "dex_%02d" % (idx)
                dec_idx_mak = Path(dec_images).joinpath(dec_idx)
                dec_gif_mak = Path(gif_dec).joinpath(dec_idx)
                os.makedirs(dec_idx_mak)
                os.makedirs(dec_gif_mak)
        elif model == "ppo":
            self.gif_dir = Path(outdir).joinpath("gif_images")
            gif_input = Path(self.gif_dir).joinpath("gif_input")
            gif_bard = Path(self.gif_dir).joinpath("gif_bard")
            gif_action = Path(self.gif_dir).joinpath("gif_action")
            gif_value = Path(self.gif_dir).joinpath("gif_value")
            self.image_dir = Path(outdir).joinpath("images")
            image_input = Path(self.image_dir).joinpath("image_input")
            bard_input = Path(self.image_dir).joinpath("bard_input")
            image_action = Path(self.image_dir).joinpath("image_action")
            image_value = Path(self.image_dir).joinpath("image_value")
            os.makedirs(self.gif_dir)
            os.makedirs(gif_input)
            os.makedirs(gif_bard)
            os.makedirs(gif_action)
            os.makedirs(gif_value)
            os.makedirs(self.image_dir)
            os.makedirs(image_input)
            os.makedirs(bard_input)
            os.makedirs(image_action)
            os.makedirs(image_value)
            for i in range(num_envs):
                in_ep = "image_%02d_input" % (i)
                bd_ep = "image_%02d_bard" % (i)
                ac_ep = "image_%02d_action" % (i)
                va_ep = "image_%02d_value" % (i)
                image_ep_input = Path(image_input).joinpath(in_ep)
                image_ep_bard = Path(bard_input).joinpath(bd_ep)
                image_ep_action = Path(image_action).joinpath(ac_ep)
                image_ep_value = Path(image_value).joinpath(va_ep)
                os.makedirs(image_ep_input)
                os.makedirs(image_ep_bard)
                os.makedirs(image_ep_action)
                os.makedirs(image_ep_value)
    def save_image(self, camera_ani_all, enc_ani_all, dec_ani_all, step, batch_size, episode, n_actions):
        ani_camera = [ [] for i in range(batch_size)]
        ani_enc = [ [] for i in range(batch_size)]
        ani_dec = [ [ [] for j in range(batch_size) ] for i in range(n_actions)]

        enc_max = enc_ani_all.max(axis=None, keepdims=True)
        enc_min = enc_ani_all.min(axis=None, keepdims=True)
        enc_ani_all = min_max(enc_ani_all, enc_min, enc_max)
        for j in range(batch_size):
            for i in range(step):
                camera = camera_ani_all[j ,i, :, :, :] * 255
                camera = cv2.cvtColor(camera, cv2.COLOR_RGB2BGR).astype(np.uint8)
                gray = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)
                gray = np.repeat(gray[..., np.newaxis], 3, axis=-1)
                camera = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
                camera = Image.fromarray(camera)
                ani_camera[j].append(camera)
                camera.save(Path(self.image_dir).joinpath(f'image_input/camera_{j:02}_{i:02}.png'))

                enc_image = enc_ani_all[j ,i, :, :] * 255
                enc_image = cv2.resize(
                    enc_image,
                    dsize=(128,128),
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.uint8)
                enc_image = cv2.applyColorMap(enc_image, cv2.COLORMAP_JET)
                enc_image = cv2.addWeighted(
                    gray, 0.7, enc_image, 0.3, 0
                )
                enc_image = cv2.cvtColor(enc_image, cv2.COLOR_BGR2RGB)
                enc_image = Image.fromarray(enc_image)
                ani_enc[j].append(enc_image)
                enc_image.save(Path(self.image_dir).joinpath(f'enc_image/enc_images{j:04}_{i:02}.png'))
                for a in range(n_actions):
                    dec_animation = dec_ani_all[a, :, :, :, :]
                    dec_max = dec_animation.max(axis=None, keepdims=True)
                    dec_min = dec_animation.min(axis=None, keepdims=True)
                    dec_animation = min_max(dec_animation, dec_min, dec_max)
                    dec_image = dec_animation[j ,i, :, :] * 255
                    dec_image = cv2.resize(
                        dec_image,
                        dsize=(128,128),
                        interpolation=cv2.INTER_LINEAR,
                    ).astype(np.uint8)
                    dec_image = cv2.applyColorMap(dec_image, cv2.COLORMAP_JET)
                    dec_image = cv2.addWeighted(
                        gray, 0.7, dec_image, 0.3, 0
                    )
                    dec_image = cv2.cvtColor(dec_image, cv2.COLOR_BGR2RGB)
                    dec_image = Image.fromarray(dec_image)
                    ani_dec[a][j].append(dec_image)
                    dec_image.save(Path(self.image_dir).joinpath(f'dec_image/dex_{j:02}/dec_images{a:04}_{i:02}.png'))
        
        for j in range(batch_size):
            ani_camera[j][0].save(Path(self.gif_dir).joinpath(f'gif_input/camera_ani_{episode:04}_{j:02}.gif') , save_all = True , append_images = ani_camera[j][1:] , duration = 400 , loop = 0)
            ani_enc[j][0].save(Path(self.gif_dir).joinpath(f'gif_enc/camera_ani_{episode:04}_{j:02}.gif') , save_all = True , append_images = ani_enc[j][1:] , duration = 400 , loop = 0)
            ani_camera[j][0].save(Path(self.gif_dir).joinpath(f'gif_input/camera_slowani_{episode:04}_{j:02}.gif') , save_all = True , append_images = ani_camera[j][1:] , duration = 1000 , loop = 0)
            ani_enc[j][0].save(Path(self.gif_dir).joinpath(f'gif_enc/camera_slowani_{episode:04}_{j:02}.gif') , save_all = True , append_images = ani_enc[j][1:] , duration = 1000 , loop = 0)
            for a in range(n_actions):
                ani_dec[a][j][0].save(Path(self.gif_dir).joinpath(f'gif_dec/dex_{j:02}/camera_ani_{episode:04}_{a:02}.gif') , save_all = True , append_images = ani_dec[a][j][1:] , duration = 400 , loop = 0)
                ani_dec[a][j][0].save(Path(self.gif_dir).joinpath(f'gif_dec/dex_{j:02}/camera_anislow_{episode:04}_{a:02}.gif') , save_all = True , append_images = ani_dec[a][j][1:] , duration = 1000 , loop = 0)
        step = 0
        print("OK!")

    def make_image(self, agent, camera_ani, enc_ani, dec_ani, camera_ani_all, enc_ani_all, dec_ani_all, batch_size, n_actions):
        camera_all = agent.model.input_image.cpu()
        camera_all = camera_all.detach().numpy()
        conv_features, enc_attn_weights, dec_attn_weights = agent.conv_features, agent.enc_attn_weights, agent.dec_attn_weights
        for i in range(batch_size):
            dec_idx_ani = []

            camera = camera_all[i]
            camera = camera.transpose(1, 2, 0)[np.newaxis, :, :, :]
            camera_ani = camera if camera_ani == [] else np.concatenate([camera_ani,camera])

            _, _, h, w = conv_features.shape
            for idx in range(n_actions):
                dec_image = dec_attn_weights[i, idx].view(h, w).to("cpu")
                dec_image = dec_image.detach().numpy()[np.newaxis, :, :]
                dec_idx_ani = dec_image if dec_idx_ani == [] else np.concatenate([dec_idx_ani,dec_image])
            dec_idx_ani = dec_idx_ani[:, np.newaxis, :, :]
            dec_ani = dec_idx_ani if dec_ani == [] else np.concatenate([dec_ani,dec_idx_ani], 1)

            sattn = enc_attn_weights[i].reshape(h, w, h, w).to("cpu")
            sattn = sattn.detach().numpy()
            all_sattn = []
            for he in range(h):
                for we in range(w):
                    all_sattn.append(sattn[he, we, :, :])
            enc_image = np.mean(np.array(all_sattn), axis=0)[np.newaxis, :, :] 
            enc_ani = enc_image if enc_ani == [] else np.concatenate([enc_ani,enc_image])

        camera_ani = camera_ani[:, np.newaxis, :, :, :]
        enc_ani = enc_ani[:, np.newaxis, :, :]
        dec_ani = dec_ani[:, :, np.newaxis, :, :]
        return camera_ani, enc_ani, dec_ani
    
    def write_mask(self, env, agent, step, model):
        global camera_ani
        global bard_ani
        global model_att_v_ani
        global model_att_a_ani
        batch_size = env.num_envs
        episode = step // env.max_episode_length
        
        if step % env.max_episode_length == 0:
            if step > 0:
                for i in range(batch_size):
                    camera_ani[i][0].save(Path(self.gif_dir).joinpath(f'gif_input/camera_ani_{episode:04}_{i:02}.gif') , save_all = True , append_images = camera_ani[i][1:] , duration = 400 , loop = 0)
                    bard_ani[i][0].save(Path(self.gif_dir).joinpath(f'gif_bard/bard_ani_{episode:04}_{i:02}.gif') , save_all = True , append_images = bard_ani[i][1:] , duration = 400 , loop = 0)
                    if model in ["mask_single_value", "mask_double", "new_mask_double", "target_mask_double"]:
                        model_att_v_ani[i][0].save(Path(self.gif_dir).joinpath(f'gif_value/model_att_v_ani_{episode:04}_{i:02}.gif') , save_all = True , append_images = model_att_v_ani[i][1:] , duration = 400 , loop = 0)
                    if model in ["mask_single_policy", "mask_double", "new_mask_double", "target_mask_double"]:
                        model_att_a_ani[i][0].save(Path(self.gif_dir).joinpath(f'gif_action/model_att_a_ani_{episode:04}_{i:02}.gif') , save_all = True , append_images = model_att_a_ani[i][1:] , duration = 400 , loop = 0)
            camera_ani = [ [] for i in range(batch_size)]
            bard_ani = [ [] for i in range(batch_size)]
            model_att_v_ani = [ [] for i in range(batch_size)]
            model_att_a_ani = [ [] for i in range(batch_size)]
        camera_all = agent.model.input_image.cpu()
        camera_all = camera_all.detach().numpy()
        bard_all = env.bard_cam_tensors
        if model in ["mask_single_value", "mask_double", "new_mask_double", "target_mask_double", "stripe_mask_double"]:
            model_att_v_all = agent.model.att_v_sig5.cpu()
            model_att_v_all = model_att_v_all.detach().numpy()
        if model in ["mask_single_policy", "mask_double", "new_mask_double", "target_mask_double", "stripe_mask_double"]:
            model_att_a_all = agent.model.att_a_sig5.cpu()
            model_att_a_all = model_att_a_all.detach().numpy()
            """model_att_a_all_inv = agent.model.att_a_sig5_inv.cpu()
            model_att_a_all_inv = model_att_a_all_inv.detach().numpy()"""

        for i in range(batch_size):
            camera = camera_all[i]
            camera = camera.transpose(1, 2, 0) * 255
            camera = cv2.cvtColor(camera, cv2.COLOR_RGB2BGR).astype(np.uint8)
            gray = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)
            gray = np.repeat(gray[..., np.newaxis], 3, axis=-1)
            camera = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
            camera = Image.fromarray(camera)
            camera.save(Path(self.image_dir).joinpath(f'image_input/image_{i:02}_input/camera_{episode:04}_{step:04}.png'))
            camera_ani[i].append(camera)

            bard = bard_all[i].cpu().numpy()
            bard = Image.fromarray(bard)
            bard.save(Path(self.image_dir).joinpath(f'bard_input/image_{i:02}_bard/bard_{episode:04}_{step:04}.png'))
            bard_ani[i].append(bard)

            if model in ["mask_single_policy", "mask_double", "new_mask_double", "target_mask_double", "stripe_mask_double"]:
                model_att_v = model_att_v_all[i]
                model_att_v = min_max_image(model_att_v)
                model_att_v = model_att_v.transpose(1, 2, 0) * 255
                model_att_v = cv2.resize(
                    model_att_v,
                    dsize=gray.shape[:2],
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.uint8)
                model_att_v_color = cv2.applyColorMap(model_att_v, cv2.COLORMAP_JET)
                model_att_v_color = cv2.addWeighted(
                    gray, 0.7, model_att_v_color, 0.3, 0
                )
                model_att_v_color = cv2.cvtColor(model_att_v_color, cv2.COLOR_BGR2RGB)
                model_att_v_color = Image.fromarray(model_att_v_color)
                model_att_v_color.save(Path(self.image_dir).joinpath(f'image_value/image_{i:02}_value/value_{episode:04}_{step:04}.png'))
                model_att_v_ani[i].append(model_att_v_color)

            if model in ["mask_single_policy", "mask_double", "new_mask_double", "target_mask_double", "stripe_mask_double"]:
                model_att_a = model_att_a_all[i]
                model_att_a = min_max_image(model_att_a)
                model_att_a = model_att_a.transpose(1, 2, 0) * 255
                model_att_a = cv2.resize(
                    model_att_a,
                    dsize=gray.shape[:2],
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.uint8)
                model_att_a_color = cv2.applyColorMap(model_att_a, cv2.COLORMAP_JET)
                model_att_a_color = cv2.addWeighted(
                    gray, 0.7, model_att_a_color, 0.3, 0
                )
                model_att_a_color = cv2.cvtColor(model_att_a_color, cv2.COLOR_BGR2RGB)
                model_att_a_color = Image.fromarray(model_att_a_color)
                model_att_a_color.save(Path(self.image_dir).joinpath(f'image_action/image_{i:02}_action/action_{episode:04}_{step:04}.png'))
                model_att_a_ani[i].append(model_att_a_color)