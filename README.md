Below is the full Markdown you can copy-paste directly into `README.md`:

```markdown
# CycleGAN for Vehicle Sim-to-Real Visual Transfer  
_A lightweight PyTorch implementation_

| **Author** | Cristian Cubides-Herrera |
|------------|--------------------------|

> A concise, reproducible CycleGAN pipeline that translates **synthetic vehicle-camera frames** (e.g. Unity, Carla, LGSVL) to **real-world road imagery**, closing the visual domain gap before reinforcement-learning or control pipelines consume the observations.

---

## 2. Quick Start

```bash
# 1. Clone and create an isolated environment
git clone https://github.com/Cubos1998/CycleGAN_DonkeyCar.git
cd CycleGAN_DonkeyCar
python -m venv venv_cycleGAN
source venv_cycleGAN/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. (Optional) grab the demo dataset
bash download_dataset.sh      # downloads Monet2Photo or sim2Car_complete

# 3. Train
python trainCyGAN.py \
        --data_root monet2photo \
        --epochs 100 \
        --batch_size 1 \
        --lr 1e-4 \
        --save_dir checkpoints \
        --log_dir runs/cyclegan_car \
        --device cuda:0 \
        --seed 42
````

### Monitoring

```bash
tensorboard --logdir runs/cyclegan_car
```

Live losses, learning-rate schedules, and example image grids are streamed every epoch.

---

## 3. Script Highlights

| Feature                     | Where                      | Notes                                                        |
| --------------------------- | -------------------------- | ------------------------------------------------------------ |
| **Reproducibility**         | `trainCyGAN.py`            | Global & worker seeds; deterministic cuDNN; DataLoader seeds |
| **Parameter count**         | `count_params()`           | Prints number of trainable parameters for each net           |
| **Step-decay LR**           | `step_decay()`             | Halves LR each epoch (configurable)                          |
| **PatchGAN Discriminator**  | `Discriminator`            | 70 × 70 patch output for stable adversarial training         |
| **Identity & Cycle losses** | `loss_idt*`, `loss_cycle*` | Weights 0.5 and 10.0 by default – preserve colour / content  |
| **Checkpoints**             | `checkpoints/*.pth`        | Four weights saved per epoch (G\_AB, G\_BA, D\_A, D\_B)      |
| **TensorBoard images**      | `writer.add_images`        | Clamped to \[0, 1] for visual sanity-checks                  |

---

## 4. Integrating with RL

1. Copy the **G\_BA** weights (`*_netG_BA_epoch_XX.pth`) into your RL workspace.

2. Load the generator in your perception stack (e.g. DonkeyCar camera callback):

   ```python
   from models import Generator
   G_BA = Generator().to(device)
   G_BA.load_state_dict(torch.load("checkpoints/CarF_netG_BA_epoch_100.pth"))
   G_BA.eval()

   def translate(frame):
       with torch.no_grad():
           tensor = preprocess(frame).unsqueeze(0).to(device)
           fake = G_BA(tensor)
           return postprocess(fake)
   ```

3. Feed `translate(obs)` into the policy network instead of raw simulation pixels.

---

## 5. Extending / Customising

* **Different resolutions** – adjust `transforms.Resize` and final `Conv2d` paddings.
* **Alternative schedulers** – plug in `torch.optim.lr_scheduler.CosineAnnealingLR`, etc.
* **Multi-GPU** – wrap models with `torch.nn.DataParallel` or migrate to `DDP`.
* **Domain-specific augmentations** – swap `RandomHorizontalFlip` for blur, noise, etc.

---

## 6. License & Citation

Released under the **MIT License**. If you use this code in academic work, cite the original CycleGAN paper[^1] and this repo:

```
@misc{cubides2025cyclegan,
  author = {Cubides-Herrera, Cristian},
  title  = {CycleGAN for Vehicle Sim-to-Real Visual Transfer},
  year   = {2025},
  howpublished = {\url{https://github.com/<your-org>/CycleGAN-Sim2Real}}
}
```

[^1]: Zhu, J.-Y. *et al.* “Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.” ICCV 2017.

---

Enjoy bridging the reality gap! For questions or pull requests, open an issue or ping **@Cubos1998**.

```
```
