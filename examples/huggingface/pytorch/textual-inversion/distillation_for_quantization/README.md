# Distillation for quantization on Textual Inversion models to personalize text2image

[Textual inversion](https://arxiv.org/abs/2208.01618) is a method to personalize text2image models like stable diffusion on your own images._By using just 3-5 images new concepts can be taught to Stable Diffusion and the model personalized on your own images_
The `textual_inversion.py` script shows how to implement the training procedure and adapt it for stable diffusion.
We have enabled distillation for quantization in `textual_inversion.py` to do quantization aware training as well as distillation on the model generated by Textual Inversion method.

## Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

```bash
pip install transformers~=4.21.0
pip install -r requirements.txt
```
>**Note**: Please use transformers no higher than 4.21.0

## Get a FP32 Textual Inversion model

Please refer to this [Textual Inversion example](https://github.com/intel/neural-compressor/blob/master/examples/pytorch/diffusion_model/diffusers/textual_inversion/README.md) in Intel® Neural Compressor for more details.

## Do distillation for quantization

Once you have the FP32 Textual Inversion model, the following command will take the FP32 Textual Inversion model as input to do distillation for quantization and generate the INT8 Textual Inversion model.

```bash
accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$FP32_MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --use_ema --learnable_property="object" \
  --placeholder_token="<dicoo>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=300 \
  --learning_rate=5.0e-04 --max_grad_norm=3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="int8_model" \
  --do_quantization --do_distillation --verify_loading
```

## Inference

Once you have trained a INT8 model with the above command, the inference can be done simply using the `text2images.py` script. Make sure to include the `placeholder_token` in your prompt.

```bash
python text2images.py \
  --pretrained_model_name_or_path=$INT8_MODEL_NAME \
  --caption "a lovely <dicoo> in red dress and hat, in the snowly and brightly night, with many brighly buildings." \
  --images_num 4
```

Here is the comparison of images generated by the FP32 model (left) and INT8 model (right) respectively:

<img src="./images/FP32.png" width = "200" height = "200" alt="FP32" align=center />
<img src="./images/INT8.png" width = "200" height = "200" alt="FP32" align=center />

