#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wandb

def main():
   for epoch_no in range(10):
       value1=epoch_no*10
       value2=epoch_no*9
       train_log = {'value1':value1, 'value2':value2}
       wandb.log(train_log, step=epoch_no, commit=True)
   print("Training finished")


if __name__ == '__main__':
    args = {'arg1':0.7,'arg2':0.8}
    wandb.init(entity="uclnlp", project="kbc_meta", group="baselines", name="hello_world7")
    wandb.config.update(args)
    main()
    wandb.run.summary.update({'value1': 0.1, 'value2': 0.2})
    wandb.save(logs/array.err)
    wandb.save(logs/array.out)
    wandb.finish()
