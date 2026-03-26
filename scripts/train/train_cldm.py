import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from my_dataset import MyDataset
from cldm.logger import ImageLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from cldm.model import create_model, load_state_dict
from cldm.hack import disable_verbosity, enable_sliced_attention
from utils.config import *

# import debugpy; debugpy.listen(('127.0.0.1', 56789)); debugpy.wait_for_client()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # limit gpu
    save_memory = False

    disable_verbosity()

    if save_memory:
        enable_sliced_attention()

    # Configs
    resume_path = os.path.join(model_root, "control_dresscode_ini.ckpt")
    batch_size = 4
    logger_freq = 4600 # val
    learning_rate = 1.0e-05
    sd_locked = True
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('configs/cldm_v2.yaml').cpu()
    if os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}")
        state_dict = load_state_dict(resume_path, location='cpu')
        # Allow missing / unexpected keys because we added new modules
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print("Missing keys when loading checkpoint (expected due to new modules):")
            print(f"  {len(missing)} keys (showing first 20): {missing[:20]}")
        if unexpected:
            print("Unexpected keys in checkpoint (ignored):")
            print(f"  {len(unexpected)} keys (showing first 20): {unexpected[:20]}")
    else:
        print(f"Checkpoint not found at {resume_path}, training from scratch.")
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    dataset = MyDataset()
    print("******************************************************")
    print(len(dataset))
    print("******************************************************")
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    # ModelCheckpoint 
    checkpoint_callback = ModelCheckpoint(
        monitor=None,  
        dirpath='./hiera_logs',  # dirpath
        filename='model_{epoch:02d}-{step:06d}',  # file_name
        save_top_k=-1,  # save all model
        save_last=True,  # save last model
        save_weights_only=False,  
        mode='min',  # Save when the validation indicator is minimized
        every_n_train_steps=50000  
    )

    # logger and ModelCheckpoint 
    callbacks = [logger, checkpoint_callback]
    trainer = pl.Trainer(gpus=[0], precision=32, callbacks=callbacks, max_epochs=100)

    # Train!
    trainer.fit(model, dataloader)
