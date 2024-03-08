from deepreg.callback import build_checkpoint_callback
from deepreg.predict import normalize_batched_image
from deepreg.registry import REGISTRY
from deepreg.util import build_dataset
import deepreg.model.optimizer as opt
import deepreg.train as train
import tensorflow as tf

config_path = "/data3/prj_register/2024-03-08-train-2I/config_batch.yaml"
log_dir = "/data3/prj_register/2024-03-08-train-2I"
ckpt_path = ""
exp_name = "centroid_labels_augmented_batched_hybrid"
max_epochs = 300

config, log_dir, ckpt_path = train.build_config(
    config_path=config_path,
    log_dir=log_dir,
    exp_name=exp_name,
    ckpt_path=ckpt_path,
    max_epochs=max_epochs,
)
config["train"]["preprocess"]["batch_size"] = 4
batch_size = config["train"]["preprocess"]["batch_size"]

data_loader_train, dataset_train, steps_per_epoch_train = build_dataset(
    dataset_config=config["dataset"],
    preprocess_config=config["train"]["preprocess"],
    split="train",
    training=True,
    repeat=True,
)

data_loader_val, dataset_val, steps_per_epoch_val = build_dataset(
    dataset_config=config["dataset"],
    preprocess_config=config["train"]["preprocess"],
    split="valid",
    training=False,
    repeat=True,
)
model: tf.keras.Model = REGISTRY.build_model(
    config=dict(
        name=config["train"]["method"],
        moving_image_size=data_loader_train.moving_image_shape,
        fixed_image_size=data_loader_train.fixed_image_shape,
        moving_label_size=(200,3),
        fixed_label_size=(200,3),
        index_size=data_loader_train.num_indices,
        labeled=config["dataset"]["train"]["labeled"],
        batch_size=batch_size,
        config=config["train"],
    )
)
optimizer = opt.build_optimizer(optimizer_config=config["train"]["optimizer"])
model.compile(optimizer=optimizer)
model.plot_model(output_dir=log_dir)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=config["train"]["save_period"],
    update_freq=config["train"].get("update_freq", "epoch"),
)
ckpt_callback, initial_epoch = build_checkpoint_callback(
    model=model,
    dataset=dataset_train,
    log_dir=log_dir,
    save_period=config["train"]["save_period"],
    ckpt_path=ckpt_path,
)
callbacks = [tensorboard_callback, ckpt_callback]
history = model.fit(
    x=dataset_train,
    steps_per_epoch=steps_per_epoch_train,
    initial_epoch=1,
    epochs=config["train"]["epochs"],
    validation_data=dataset_val,
    validation_steps=steps_per_epoch_val,
    callbacks=callbacks,
)
