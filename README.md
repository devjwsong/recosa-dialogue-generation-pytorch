# recosa-dialogue-generation-pytorch

This is a multi-turn chatbot project using the **ReCoSa** structure introduced in *ReCoSa: Detecting the Relevant Contexts with Self-Attention for
Multi-turn Dialogue Generation*[[1]](#1).

The model detects the relevant dialogue histories with the self-attention mechanism, which uses the history-level transformer encoder, not the word-level.

The details of structure are as follows.

<img src="https://user-images.githubusercontent.com/16731987/97245796-2d7b6580-183f-11eb-9560-0c36038c0124.png" alt="The description of the ReCoSa structure." style="width: 60%; margin-left: 0;">

<br/>

---

### Arguments

**Arguements for training**

| Argument              | Type              | Description                                                  | Default               |
| --------------------- | ----------------- | ------------------------------------------------------------ | --------------------- |
| `seed` | `int` | The random seed number for training. | `0` |
| `data_dir`          | `str`       | The name of the parent directory where the whole data files are stored. | `"data"`              |
| `task`     | `str`       | The name of the specific task(dataset) name. (`"daily_dialog"`, `"empathetic_dialogues"`, `"persona_chat"`, `"blended_skill_talk"`) | *YOU MUST SPECIFY* |
| `pad_token`         | `str`        | The pad token.                             | `"<pad>"`             |
| `bos_token`         | `str`       | The bos token.          | `"<bos>"`             |
| `eos_token`         | `str`       | The eos token.                | `"<eos>"`             |
| `sp1_token` | `str` | The speaker1 token. | `"<sp1>"` |
| `sp2_token` | `str` | The speaker2 token. | `"<sp2>"` |
| `learning_rate`     | `float` | The initial learning rate.                                   | `5e-4`                |
| `warmup_ratio` | `float` | The warmup step ratio. | `0.0` |
| `max_grad_norm` | `float` | The max value for gradient clipping. | `1.0` |
| `train_batch_size` | `int` | The batch size for training.                   | `32`              |
| `eval_batch_size` | `int` | The batch size for evaluation. | `8` |
| `num_workers` | `int` | The number of workers for data loading. | `0` |
| `num_epochs`        | `int` | The number of training epochs. | `10`              |
| `src_max_len` | `int` | The max length of each input utterance. | `128`                 |
| `max_turns` | `int` | The max number of utterances to be included. | `10` |
| `trg_max_len` | `int` | The max length of a target response. | `128` |
| `num_heads`         | `int` | The number of heads for multi-head attention. | `8`                   |
| `num_encoder_layers` | `int` | The number of layers in the utterance-level encoder. | `6`                   |
| `num_gru_layers` | `int` | The number of layers in the word-level encoder. | `2` |
| `gru_dropout` | `float` | The dropout rate of the word-level encoder. | `0.1` |
| `num_decoder_layers` | `int` | The number of layers in the decoder. | `2`                  |
| `d_model`           | `int` | The hidden size inside of the transformer module. | `768`              |
| `d_pos` | `int` | The hidden size of the positional embedding. | `256` |
| `d_ff`              | `int` | The intermediate hidden size of each feed-forward layer. | `2048`                |
| `dropout`           | `int` | The dropout rate of the transformer modules. | `0.1`                 |
| `gpus` | `str` | The indices of GPUs to use. (This should be a string which contains index values separated with commas. ex: `"0, 1, 2, 3"`) | `"0"` |
| `num_nodes` | `int` | The number of machine. | `1` |

<br/>

**Arguments for inference**

| Argument        | Type    | Description                                                  | Default            |
| --------------- | ------- | ------------------------------------------------------------ | ------------------ |
| `pad_token`   | `str`   | The pad token.                                               | `"<pad>"`          |
| `bos_token`   | `str`   | The bos token.                                               | `"<bos>"`          |
| `eos_token`   | `str`   | The eos token.                                               | `"<eos>"`          |
| `sp1_token`   | `str`   | The speaker1 token.                                          | `"<sp1>"`          |
| `sp2_token`   | `str`   | The speaker2 token.                                          | `"<sp2>"`          |
| `src_max_len` | `int`   | The max length of each input utterance.                      | `128`              |
| `max_turns`   | `int`   | The max number of utterances to be included.                 | `10`               |
| `trg_max_len` | `int`   | The max length of a target response.                         | `128`              |
| `gpus`        | `str`   | The indices of GPUs to use. (When inferencing, only a single GPU is used. If you try to set mutiple GPUs, the assertion error will be raised.) | `"0"`              |
| `top_p`       | `float` | The top-p value for nucleus sampling decoding.               | `0.9`              |
| `end_command` | `str`   | The command to stop the conversation when inferencing.       | `"Abort!"`         |
| `log_idx`     | `int`   | The index of a lightning log directory which contains the checkpoints to use. | *YOU MUST SPECIFY* |
| `ckpt_file`   | `str`   | The full name of the trained checkpoint for inference.       | *YOU MUST SPECIFY* |

<br/>

<hr style="background: transparent; border: 0.5px dashed;"/>

### Datasets

By default, I propose the codes for downloading the datasets and preprocessing.

There are 4 types of the default datasets as follows.

<br/>

- DailyDialog[[2]](#2)
- EmpatheticDialogues[[3]](#3)
- Persona-Chat[[4]](#4)
- BlendedSkillTalk[[5]](#5)

<br/>

For this project, we use the ParlAI[[6]](#6) platform made by Facebook, to download the datasets we need.

This repository also provides a useful parsing script for each downloaded data.

The detailed instruction for using ParlAI can be found in the official document[[7]](#7).

<br/>

<hr style="background: transparent; border: 0.5px dashed;"/>

### How to run

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Clone the official ParlAI repository in your project directory.

   ```shell
   git clone https://github.com/facebookresearch/ParlAI.git && cd ParlAI
   ```

   <br/>

3. Setup ParlAI and download the data.

   ```shell
   python setup.py develop
   parlai display_data --task dailydialog
   parlai display_data --task empathetic_dialogues
   parlai display_data --task personachat
   parlai display_data --task blended_skill_talk
   cd ..
   ```

   ParlAI has a lot of useful dialogue corpus beside 4 datasets mentioned above.

   You can check the list of the tasks it supports in the document.

   <br/>

4. Parse each data and save them info `*.pickle` and `*.json` files. (After parsing, you can delete ParlAI repo.)

   ```shell
   python src/parse_data.py --data_dir=DATA_DIR
   ```

   - `--data_dir`: The name of the parent directory where the whole data files are stored. (default: `"data"`)

   <br/>

5. Run the following command to train the model.

   ```shell
   sh exec_train.sh
   ```

   <br/>

6. Run below command to conduct an inference with the trained model.

   ```shell
   sh exec_infer.sh
   ```


<br/>

---

### References

<a id="1">[1]</a> Zhang, H., Lan, Y., Pang, L., Guo, J., & Cheng, X. (2019). Recosa: Detecting the relevant contexts with self-attention for multi-turn dialogue generation. *arXiv preprint arXiv:1907.05339*. ([https://arxiv.org/abs/1907.05339](https://arxiv.org/abs/1907.05339))

<a id="2">[2]</a> Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). Dailydialog: A manually labelled multi-turn dialogue dataset. *arXiv preprint arXiv:1710.03957*. ([https://arxiv.org/abs/1710.03957](https://arxiv.org/abs/1710.03957))

<a id="3">[3]</a> Rashkin, H., Smith, E. M., Li, M., & Boureau, Y. L. (2018). Towards empathetic open-domain conversation models: A new benchmark and dataset. *arXiv preprint arXiv:1811.00207*. ([https://arxiv.org/abs/1811.00207](https://arxiv.org/abs/1811.00207))

<a id="4">[4]</a> Zhang, S., Dinan, E., Urbanek, J., Szlam, A., Kiela, D., & Weston, J. (2018). Personalizing dialogue agents: I have a dog, do you have pets too?. *arXiv preprint arXiv:1801.07243*. ([https://arxiv.org/abs/1801.07243](https://arxiv.org/abs/1801.07243))

<a id="5">[5]</a> Smith, E. M., Williamson, M., Shuster, K., Weston, J., & Boureau, Y. L. (2020). Can You Put it All Together: Evaluating Conversational Agents' Ability to Blend Skills. *arXiv preprint arXiv:2004.08449*. ([https://arxiv.org/abs/2004.08449](https://arxiv.org/abs/2004.08449))

<a id="6">[6]</a> Miller, A. H., Feng, W., Fisch, A., Lu, J., Batra, D., Bordes, A., ... & Weston, J. (2017). Parlai: A dialog research software platform. *arXiv preprint arXiv:1705.06476*. ([https://arxiv.org/abs/1705.06476](https://arxiv.org/abs/1705.06476))

<a id="7">[7]</a> https://parl.ai/docs/index.html

