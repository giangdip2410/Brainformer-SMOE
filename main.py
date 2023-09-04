# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3

import math
import time
import random
import numpy as np
import torch
import datetime
import itertools

from config import PARAMS_CONFIG
from data import get_train_val_test_data
from models import TransformerSeq
from trainer import train_iteration, full_eval
from utils import (
    get_params,
    set_up_env,
    get_optimizer_and_scheduler,
    load_checkpoint,
    save_checkpoint,
    create_exp_dir,
    Logger,
)
from tqdm import tqdm
from new_utils import train_gate, args_moe, adjust_moe_gate_number


def launch(
    env_params,
    model_params,
    adapt_span_params,
    optim_params,
    data_params,
    trainer_params,
    args_moe=args_moe,
):
    # ENVIRONMENT (device, distributed, etc.)
    global train_step, current_gate, best_val_loss
    set_up_env(env_params)
    device = env_params["device"]
    distributed = env_params["distributed"]

    if distributed == False or env_params["rank"] == 0:
        print("model_params:\t", model_params)
        print("optim_params:\t", optim_params)
        print("data_params:\t", data_params)
        print("trainer_params:\t", trainer_params)
        print("adapt_span_params:\t", adapt_span_params)

    # DATA
    train_data, val_data, test_data = get_train_val_test_data(
        data_params=data_params,
        env_params=env_params,
        batch_size=trainer_params["batch_size"],
        device=device,
    )
    # MODEL
    model = TransformerSeq(
        vocab_size=data_params["vocab_size"],
        **model_params,
        adapt_span_params=adapt_span_params,
    )
    if distributed:
        local_rank = env_params["local_rank"]
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    else:
        model = torch.nn.DataParallel(model)
        model = model.to(device)

    # OPTIMIZER AND SCHEDULER
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, optim_params=optim_params
    )
    # Freeze gate
    if model_params["policy"] != "smoe":
        train_gate(model, train_gate=False)
    # create logger
    logger = Logger()
    fold_name = trainer_params["checkpoint_path"].split("/")[-1].split(".")[0]
    logging = create_exp_dir(f"experiments/{fold_name}")
    # log paramters
    logging(f"Training Parameters:\n {trainer_params}")
    # logging time
    current_time = datetime.datetime.now()
    logging(str(current_time))
    # log model
    logging(str(model))
    logging(f"Total of Prams: {sum(p.numel() for p in model.parameters())}")
    logging(
        f"Total of Trainable Prams: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # resume training from last checkpoint if exists
    iter_init = load_checkpoint(
        trainer_params["checkpoint_path"],
        model,
        optimizer,
        scheduler,
        logger,
        distributed,
    )
    # cur_step = 0
    start_time = time.time()
    if trainer_params["full_eval_mode"]:
        # evaluate the model on test data
        with torch.no_grad():
            loss_val = full_eval(
                model,
                optimizer,
                scheduler,
                val_data,
                model_params["block_size"],
                model_params["hidden_size"],
                cur_step=train_step,
                logging=logging,
            )
            loss_test = full_eval(
                model,
                optimizer,
                scheduler,
                test_data,
                model_params["block_size"],
                model_params["hidden_size"],
                cur_step=train_step,
                logging=logging,
            )
            if distributed:
                # collect results into rank0
                stats = torch.tensor([loss_val, loss_test]).to(device)
                torch.distributed.reduce(stats, 0)
                if env_params["rank"] == 0:
                    loss_val = stats[0] / env_params["world_size"]
                    loss_test = stats[1] / env_params["world_size"]
                else:
                    return
    
            if  ('enwik8' in data_params['data_path']) or ('text8' in data_params['data_path']):
                logging("Val: {:.3f} BPC".format(loss_val / math.log(2)))
                logging("Test: {:.3f} BPC".format(loss_test / math.log(2)))
            else:
                logging("Val: {:.3f} PPL".format(math.exp(loss_val)))
                logging("Test: {:.3f} PPL".format(math.exp(loss_test)))

        return

    # position of current batch
    data_pos = [0] * 2
    # initialize caches for train and valid
    hid_cache = [
        [
            torch.zeros(
                train_data.size(0),
                model.module.layers[layer_i].attn.attn.get_cache_size(),
                model_params["hidden_size"],
            ).to(device)
            for layer_i in range(model.module.attn_layer_count)
        ]
        for _ in range(2)
    ]
    # config
    nb_batches_per_iter = trainer_params["nb_batches_per_iter"]
    args_moe.dynamic_overall_steps = trainer_params["maxstep"]
    max_epoch = int(trainer_params["maxstep"] / (nb_batches_per_iter *trainer_params["batch_split"]) )
    current_gate = 2  # select 2 topk
    # training
    for epoch in tqdm(range(iter_init, max_epoch)):
        t_sta = time.time()
        if model_params["policy"] != "smoe":
            current_gate = adjust_moe_gate_number(
                model, train_step, args_moe, current_gate
            )
        loss_train, data_pos[0], hid_cache[0], train_step = train_iteration(
            model,
            optimizer,
            scheduler,
            train_data,
            nb_batches_per_iter,
            model_params["block_size"],
            False,
            data_pos[0],
            hid_cache[0],
            trainer_params["batch_split"],
            train_step,
            logging,
            cnt_step=True,
        )

        elapsed = 1000 * (time.time() - t_sta) / nb_batches_per_iter
        with torch.no_grad():
            loss_val, data_pos[1], hid_cache[1], _ = train_iteration(
                model,
                optimizer,
                scheduler,
                val_data,
                nb_batches_per_iter,
                model_params["block_size"],
                True,
                data_pos[1],
                hid_cache[1],
                trainer_params["batch_split"],
                cur_step=train_step,
                logging=logging,
                cnt_step=False,
            )

        if distributed:
            # collect results into rank0
            stats = torch.tensor([loss_train, loss_val]).to(device)
            torch.distributed.reduce(stats, 0)
            if env_params["rank"] == 0:
                loss_train = stats[0] / env_params["world_size"]
                loss_val = stats[1] / env_params["world_size"]
            else:
                continue
        logging(f"=================== EPOCHS {epoch} ======================")
        if  ('enwik8' in data_params['data_path']) or ('text8' in data_params['data_path']):
            logging(
                f"Epochs: {epoch} | Steps: {train_step} | Batchs: {nb_batches_per_iter}| loss_train: {loss_train} | loss_val: {loss_val} | bpc_train: {loss_train / math.log(2)} | bpc_val: {loss_val / math.log(2)} | elapsed: {elapsed}"
            )
        else:
            logging(
                f"Epochs: {epoch} | Steps: {train_step} | Batchs: {nb_batches_per_iter}| loss_train: {loss_train} | loss_val: {loss_val} | ppl_train: {math.exp(loss_train)} | ppl_val: {math.exp(loss_val)} | elapsed: {elapsed}"
            )
        logger.log_iter(
            epoch, nb_batches_per_iter, loss_train, loss_val, elapsed, model
        )
        # Save the model if the validation loss is the best we've seen so far.
        if (best_val_loss is None) or loss_val < best_val_loss:
            best_val_loss = loss_val
            save_checkpoint(
                trainer_params["checkpoint_path"],
                epoch,
                model,
                optimizer,
                scheduler,
                logger,
            )
        if train_step == trainer_params["maxstep"]:
            break
    end_time = time.time()
    logging(f"Training time total: {(end_time - start_time)/3600} h")


if __name__ == "__main__":
    # Loop over epochs.
    train_step = 0
    best_val_loss = None
    launch(**get_params(params_config=PARAMS_CONFIG))
